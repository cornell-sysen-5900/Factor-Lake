"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s09_validate.py
PURPOSE: Stage s09_validate: final validation of processed/final/ against independent sources, for teammates.

In:  processed/final/ (never written), raw/ (benchmarks, Fama-French, the raw CSVs for the recompute);
     the old notebook panel for the old-vs-new backtest (config.OLD_PANEL_PATHS, first one found)
Out: processed/s09_validation/summary.json, VALIDATION.md, recompute_sample.csv, recompute_raw.csv,
     upload_wrds.py (extracted from final/UPLOAD.md for the offline dry run)

Checks (all WARN, pipeline/PLAN.md s09_validate):
    1. index tracking: universe at t weighted by mkt_cap_musd, next-month returns, vs IWM and the Russell 2000
    2. factor sanity: value-weighted top 30% minus bottom 30% vs HML, UMD, RMW, CMA
    3. old vs new backtest: top 10% by ROA each June, equal weight, vs IWM, on the new and the old panel
    4. independent recompute of 100 random panel rows from raw CSVs (scripts/wrds_independent_recompute.py,
       which imports no pipeline code)
    5. reproducibility: every stage rerun into a temporary folder, final files compared by rows and data hash
    6. app memory of factor_panel_monthly, full and with 9 columns
    7. UPLOAD.md dry run offline: every S3 key without credentials or network
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import dictionary
from pipeline.checks import PASS, WARN, Check, check
from pipeline.config import ETF_PERMNOS, INDEX_TICS, OLD_PANEL_PATHS, DataPaths
from pipeline.outputs import fresh_stage_dir, write_summary
from pipeline.stages import STAGES

STAGE = "s09_validate"
SEED = 20261025
N_SAMPLE = 100
REL_TOL = 1e-6
APP_MEMORY_MB = 1024                 # Streamlit free tier
APP_COLUMNS = ["date", "permno", "ticker", "sector_name", "f_book_to_price", "f_earnings_to_price", "f_roa",
               "mom_12_1", "f_forward_ep"]
FACTOR_TESTS = (  # name, panel column, Fama-French factor, low minus high, WARN below this correlation
    ("value (B/P)", "f_book_to_price", "hml", False, 0.5),
    ("momentum (12-1)", "mom_12_1", "umd", False, 0.7),
    ("profitability (gross profit / assets)", "f_gross_profitability", "rmw", False, 0.5),
    ("investment (asset growth, low minus high)", "f_asset_growth_1yr", "cma", True, 0.5),
)
INFO_FACTOR_TESTS = (("profitability (ROE), informational", "f_roe", "rmw", False, None),)
RECOMPUTE = {  # script column -> (final file, column)
    "b_p": ("factor_panel_monthly.parquet", "f_book_to_price"), "roa": ("factor_panel_monthly.parquet", "f_roa"),
    "mom_12_1": ("factor_panel_monthly.parquet", "mom_12_1"),
    "e_p": ("factor_panel_monthly.parquet", "f_earnings_to_price"),
    "ev_ebitda": ("rv_multiples_monthly.parquet", "ev_ebitda"), "ret_m": ("returns_monthly.parquet", "ret_m")}
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "wrds_independent_recompute.py"


# ---------- Benchmarks from raw ----------

def iwm_monthly(raw: Path) -> pd.Series:
    etf = pd.read_csv(raw / "benchmark_etfs.csv")
    etf.columns = etf.columns.str.lower()
    iwm = etf[etf["permno"] == next(p for p, n in ETF_PERMNOS.items() if n == "iwm")]
    month = pd.to_datetime(iwm["dlycaldt"]).dt.to_period("M")
    return ((1 + iwm["dlyret"]).groupby(month).prod(min_count=1) - 1).rename("iwm")


def r2000_monthly(raw: Path) -> pd.Series:
    idx = pd.read_csv(raw / "index_close.csv", parse_dates=["datadate"])
    tic = next(t for t, n in INDEX_TICS.items() if n == "r2000")
    lvl = idx[idx["tic"] == tic].sort_values("datadate")
    end = lvl.groupby(lvl["datadate"].dt.to_period("M"))["prccd"].last()
    return end.pct_change().iloc[1:].rename("r2000")


def ff_monthly(raw: Path) -> pd.DataFrame:
    ff = pd.read_csv(raw / "ff_monthly.csv", parse_dates=["dateff"])
    ff.columns = ff.columns.str.lower()
    return ff.set_index(ff["dateff"].dt.to_period("M")).drop(columns="dateff")


# ---------- Portfolios ----------

def next_returns(panel: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Panel rows with the stock's return in the month after t (returns_monthly, delisting returns included)."""
    r = rets.assign(hold=rets["month_end"].dt.to_period("M"))[["permno", "hold", "ret_m"]]
    p = panel.assign(hold=panel["date"].dt.to_period("M") + 1)
    return p.merge(r, on=["permno", "hold"], how="left")


def weighted(m: pd.DataFrame, w: str) -> pd.Series:
    m = m.dropna(subset=["ret_m", w])
    m = m[m[w] > 0]
    return (m["ret_m"] * m[w]).groupby(m["hold"]).sum() / m[w].groupby(m["hold"]).sum()


def yearly(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


def tracking(port: pd.Series, bench: pd.Series) -> dict:
    both = pd.concat([port.rename("p"), bench.rename("b")], axis=1).dropna()
    d = both["p"] - both["b"]
    yp, yb = yearly(both["p"]), yearly(both["b"])
    return {"months": len(both), "first": str(both.index.min()), "last": str(both.index.max()),
            "correlation": round(float(both["p"].corr(both["b"])), 4),
            "tracking_error_annual": round(float(d.std() * np.sqrt(12)), 4),
            "mean_annual_difference": round(float(d.mean() * 12), 4),
            "by_year": {int(y): {"portfolio": round(float(yp[y]), 4), "benchmark": round(float(yb[y]), 4),
                                 "difference_pp": round(float((yp[y] - yb[y]) * 100), 2)} for y in yp.index}}


def contributors(m: pd.DataFrame, year: int, n: int = 3) -> list[dict]:
    """Stocks that most separate the market-cap-weighted from the IWM-weighted portfolio in a year:
    sum over months of (cap weight - IWM weight) x return."""
    m = m[(m["hold"].dt.year == year)].dropna(subset=["ret_m", "mkt_cap_musd", "index_weight"])
    m = m.assign(wc=m["mkt_cap_musd"] / m.groupby("hold")["mkt_cap_musd"].transform("sum"),
                 wi=m["index_weight"] / m.groupby("hold")["index_weight"].transform("sum"))
    c = ((m["wc"] - m["wi"]) * m["ret_m"]).groupby([m["permno"], m["ticker"]]).sum().sort_values()
    pick = pd.concat([c.head(n), c.tail(n)])
    return [{"permno": int(p), "ticker": str(t), "contribution_pp": round(float(v * 100), 2)}
            for (p, t), v in pick.items()]


def check_tracking(panel, rets, raw) -> tuple[list[Check], dict]:
    m = next_returns(panel[["date", "permno", "ticker", "mkt_cap_musd", "index_weight", "snapshot_report_dt"]], rets)
    cap, iwmw = weighted(m, "mkt_cap_musd"), weighted(m, "index_weight")
    iwm, r2k = iwm_monthly(raw), r2000_monthly(raw)
    res = {"vs_iwm": tracking(cap, iwm), "vs_r2000_price_index": tracking(cap, r2k),
           "iwm_weighted_vs_iwm": tracking(iwmw, iwm)}
    diff = (cap - iwm).dropna()
    snap = panel.groupby("date").agg(weight=("index_weight", "sum"), report=("snapshot_report_dt", "first"))
    snap["age"] = (snap.index - snap["report"]).dt.days
    by_year = snap.groupby(snap.index.year).agg(weight=("weight", "mean"), age=("age", "median"))
    off = {y: v for y, v in res["vs_iwm"]["by_year"].items() if abs(v["difference_pp"]) > 2}
    res["years_off_by_more_than_2pp"] = {
        y: {"difference_pp": v["difference_pp"],
            "iwm_weighted_difference_pp": res["iwm_weighted_vs_iwm"]["by_year"].get(y, {}).get("difference_pp"),
            # the snapshot in use is 1-2 months old, so July-August still hold the pre-reconstitution members
            "july_august_pp": round(float(diff[(diff.index.year == y) & diff.index.month.isin([7, 8])].sum() * 100), 2),
            "other_months_pp": round(float(diff[(diff.index.year == y) & ~diff.index.month.isin([7, 8])].sum() * 100),
                                     2),
            "months": int((diff.index.year == y).sum()),
            "iwm_weight_covered_pct": round(float(by_year.loc[y, "weight"]), 1) if y in by_year.index else None,
            "snapshot_age_days_median": float(by_year.loc[y, "age"]) if y in by_year.index else None,
            "top_contributors_cap_vs_iwm_weight": contributors(m, y)} for y, v in off.items()}
    t = res["vs_iwm"]
    checks = [check("index tracking: monthly correlation with IWM >= 0.99", t["correlation"] >= 0.99,
                    f"{t['correlation']:.4f} over {t['months']} months", level=WARN),
              check("index tracking: annualized tracking error vs IWM <= 2%", t["tracking_error_annual"] <= 0.02,
                    f"{t['tracking_error_annual']:.2%}; mean difference {t['mean_annual_difference']:+.2%} a year; "
                    f"{len(off)} years off by more than 2 pp", level=WARN)]
    return checks, res


def long_short(m: pd.DataFrame, col: str, low_minus_high: bool) -> pd.Series:
    x = m.dropna(subset=[col, "mkt_cap_musd", "ret_m"])
    x = x[x["mkt_cap_musd"] > 0]
    pct = x.groupby("date")[col].rank(pct=True)
    top, bottom = weighted(x[pct > 0.7], "mkt_cap_musd"), weighted(x[pct <= 0.3], "mkt_cap_musd")
    return (bottom - top) if low_minus_high else (top - bottom)


def check_factors(panel, rets, raw) -> tuple[list[Check], dict]:
    cols = list({c for _, c, *_ in (*FACTOR_TESTS, *INFO_FACTOR_TESTS)})
    m = next_returns(panel[["date", "permno", "mkt_cap_musd", *cols]], rets)
    ff = ff_monthly(raw)
    checks, res = [], {}
    for name, col, fac, lmh, lo in (*FACTOR_TESTS, *INFO_FACTOR_TESTS):
        ls = long_short(m, col, lmh)
        both = pd.concat([ls.rename("ls"), ff[fac]], axis=1).dropna()
        corr = float(both["ls"].corr(both[fac]))
        mean, sd = float(both["ls"].mean()), float(both["ls"].std())
        res[name] = {"column": col, "fama_french": fac, "months": len(both), "correlation": round(corr, 3),
                     "mean_monthly": round(mean, 5), "t_stat": round(mean / (sd / np.sqrt(len(both))), 2),
                     "fama_french_mean_monthly": round(float(both[fac].mean()), 5)}
        if lo is not None:
            checks.append(check(f"factor sanity: {name} long-short vs {fac.upper()} correlation >= {lo}",
                                corr >= lo, f"{corr:.3f}; long-short mean {mean:+.2%} a month, t = "
                                f"{res[name]['t_stat']:.2f}", level=WARN))
    return checks, res


# ---------- Old vs new backtest ----------

def june_top_roa(panel: pd.DataFrame, rets: pd.DataFrame | None = None) -> tuple[pd.Series, dict]:
    """Top 10% by f_roa at each June month end, equal weight, held July to June. With rets: every pick's
    returns_monthly return, also after it leaves the index; without: the panel's ret_next_1m while it is in it."""
    p = panel.dropna(subset=["f_roa"])
    june = p[p["date"].dt.month == 6]
    picks = {d.year: set(g.loc[g["f_roa"].rank(pct=True) > 0.9, "permno"]) for d, g in june.groupby("date")}
    form = np.where(panel["date"].dt.month >= 6, panel["date"].dt.year, panel["date"].dt.year - 1)
    if rets is None:
        x = panel.assign(form=form, hold=panel["date"].dt.to_period("M") + 1).rename(columns={"ret_next_1m": "r"})
    else:
        r = rets.assign(hold=rets["month_end"].dt.to_period("M"))
        r = r[r["hold"] >= pd.Period(f"{min(picks)}-07", "M")]
        x = r.assign(form=np.where(r["hold"].dt.month >= 7, r["hold"].dt.year, r["hold"].dt.year - 1)).rename(
            columns={"ret_m": "r"})
    sel = pd.Series([p in picks.get(f, ()) for p, f in zip(x["permno"], x["form"])], index=x.index)
    x = x[sel].dropna(subset=["r"])
    return x.groupby("hold")["r"].mean(), picks


def check_backtest(paths: DataPaths, panel, rets, raw) -> tuple[list[Check], dict]:
    old_path = next((paths.root / p for p in OLD_PANEL_PATHS if (paths.root / p).is_file()), None)
    iwm = iwm_monthly(raw)
    new_a, picks_new = june_top_roa(panel[["date", "permno", "f_roa", "ret_next_1m"]])
    new_b, _ = june_top_roa(panel[["date", "permno", "f_roa"]], rets)
    if old_path is None:
        return [check("old vs new backtest: old panel found", False, f"none of {OLD_PANEL_PATHS}", level=WARN)], {}
    old = pd.read_parquet(old_path, columns=["date", "permno", "f_roa", "ret_next_1m"])
    old_a, picks_old = june_top_roa(old)
    months = new_a.index.intersection(old_a.index).intersection(iwm.index)
    y = {k: yearly(s.reindex(months)) for k, s in (("new", new_a), ("old", old_a), ("new_full_holding", new_b),
                                                    ("iwm", iwm))}
    by_year = {int(yr): {k: round(float(v[yr]), 4) for k, v in y.items()} for yr in y["new"].index}
    for yr, v in by_year.items():
        v["new_minus_old_pp"] = round((v["new"] - v["old"]) * 100, 2)
    off = {yr: v for yr, v in by_year.items() if abs(v["new_minus_old_pp"]) > 3}
    explain = {}
    for yr in off:
        forms = sorted({yr - 1, yr} & set(picks_new) & set(picks_old))
        explain[yr] = {f"june_{f}": {"both": len(picks_new[f] & picks_old[f]),
                                     "new_only": len(picks_new[f] - picks_old[f]),
                                     "old_only": len(picks_old[f] - picks_new[f])} for f in forms}
    overlap = [len(picks_new[f] & picks_old[f]) / max(len(picks_new[f] | picks_old[f]), 1)
               for f in picks_new if f in picks_old]
    res = {"old_panel": str(old_path.relative_to(paths.root)), "months": len(months),
           "method": "June month end: top 10% by f_roa among the panel's stocks; equal weight; held July-June; "
                     "new/old: the panel's own next-month return while the stock is in the index; "
                     "new_full_holding: returns_monthly for every pick for the whole year",
           "annualized": {k: round(float((1 + s.reindex(months)).prod() ** (12 / len(months)) - 1), 4)
                          for k, s in (("new", new_a), ("old", old_a), ("new_full_holding", new_b), ("iwm", iwm))},
           "mean_june_overlap_jaccard": round(float(np.mean(overlap)), 3), "by_year": by_year,
           "years_more_than_3pp_apart": explain}
    checks = [check("old vs new backtest: no year more than 3 pp apart", not off,
                    f"{len(off)} years: {sorted(off)}; annualized new {res['annualized']['new']:.2%}, old "
                    f"{res['annualized']['old']:.2%}, IWM {res['annualized']['iwm']:.2%}", level=WARN)]
    return checks, res


# ---------- Independent recompute ----------

def check_recompute(paths: DataPaths, out: Path, final: Path) -> tuple[list[Check], dict]:
    panel = pd.read_parquet(final / "factor_panel_monthly.parquet")
    rng = np.random.default_rng(SEED)
    rows = panel.iloc[np.sort(rng.choice(len(panel), N_SAMPLE, replace=False))]
    rows[["permno", "date"]].to_csv(out / "recompute_sample.csv", index=False)
    started = time.time()
    proc = subprocess.run([sys.executable, str(SCRIPT), str(paths.raw), str(out / "recompute_sample.csv"),
                           str(out / "recompute_raw.csv")], capture_output=True, text=True)
    if proc.returncode != 0:
        return [check("independent recompute ran", False, proc.stderr[-500:], level=WARN)], {}
    got = pd.read_csv(out / "recompute_raw.csv", parse_dates=["date", "fund_period_end"])
    rv = pd.read_parquet(final / "rv_multiples_monthly.parquet", columns=["date", "permno", "ev_ebitda"])
    rets = pd.read_parquet(final / "returns_monthly.parquet", columns=["month_end", "permno", "ret_m"])
    ref = rows.merge(rv, on=["date", "permno"], how="left").merge(
        rets.rename(columns={"month_end": "date"}), on=["date", "permno"], how="left")
    m = ref.merge(got, on=["permno", "date"], how="left", suffixes=("", "_raw"))
    res, mismatches = {"seconds": round(time.time() - started), "rows": len(m), "metrics": {}}, []
    for raw_col, (_, col) in RECOMPUTE.items():
        a, b = m[col].astype(float), m[raw_col if raw_col != col else f"{col}_raw"].astype(float)
        same = (a.isna() & b.isna()) | ((a - b).abs() <= REL_TOL * np.maximum(a.abs(), b.abs()) + 1e-12)
        res["metrics"][col] = {"compared": len(m), "match": int(same.sum()), "both_missing": int((a.isna() & b.isna()).sum())}
        for i in m.index[~same]:
            r = m.loc[i]
            reason = ("different fundamentals quarter" if r["fund_period_end"] != r["fund_period_end_raw"]
                      else "company market cap differs (Compustat share flag: panel "
                           f"{bool(r['company_cap_from_compustat'])}, raw {r['compustat_flag']})"
                      if not np.isclose(r["company_mkt_cap_musd"], r["company_mkt_cap_musd_raw"], rtol=REL_TOL)
                      else "unexplained")
            mismatches.append({"permno": int(r["permno"]), "date": str(r["date"])[:10], "ticker": r["ticker"],
                               "metric": col, "panel": None if pd.isna(a[i]) else float(a[i]),
                               "raw": None if pd.isna(b[i]) else float(b[i]), "reason": reason})
    res["mismatches"] = mismatches
    n = sum(v["compared"] for v in res["metrics"].values())
    ok = sum(v["match"] for v in res["metrics"].values())
    unexplained = sum(x["reason"] == "unexplained" for x in mismatches)
    return [check(f"independent recompute: {N_SAMPLE} random rows match within {REL_TOL:g} relative", not mismatches,
                  f"{ok} of {n} values match; {len(mismatches)} differ ({unexplained} unexplained)",
                  level=WARN)], res


# ---------- Reproducibility ----------

def table_hash(path: Path, key: list[str]) -> tuple[int, str]:
    df = pd.read_parquet(path)
    df = df.sort_values(key, kind="stable").reset_index(drop=True)
    digest = hashlib.sha256(json.dumps({c: str(t) for c, t in df.dtypes.items()}).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return len(df), digest.hexdigest()


def _without(d: dict, *keys) -> dict:
    return {k: (_without(v, *keys) if isinstance(v, dict) else v) for k, v in d.items() if k not in keys}


def compare_finals(a: Path, b: Path) -> dict:
    """Every Parquet file by rows and a hash of the data sorted by its key; the JSON and Markdown files as text
    (created_at ignored)."""
    out = {}
    for f in sorted(a.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(a).as_posix()
        g = b / rel
        if not g.is_file():
            out[rel] = {"same": False, "detail": "missing in the rerun"}
        elif f.suffix == ".parquet":
            name = "prices_daily/year=YYYY/data.parquet" if rel.startswith("prices_daily/") else rel
            key = dictionary.FILES[name]["key"]
            (n1, h1), (n2, h2) = table_hash(f, key), table_hash(g, key)
            out[rel] = {"same": n1 == n2 and h1 == h2, "rows": n1, "rows_rerun": n2, "hash": h1[:16],
                        "hash_rerun": h2[:16]}
        elif f.suffix == ".json":
            j1, j2 = (_without(json.loads(x.read_text(encoding="utf-8")), "created_at") for x in (f, g))
            out[rel] = {"same": j1 == j2}
        else:
            out[rel] = {"same": f.read_text(encoding="utf-8") == g.read_text(encoding="utf-8")}
    for g in b.rglob("*"):
        if g.is_file() and not (a / g.relative_to(b)).is_file():
            out[g.relative_to(b).as_posix()] = {"same": False, "detail": "only in the rerun"}
    return out


def check_reproducible(paths: DataPaths, out: Path) -> tuple[list[Check], dict]:
    if os.environ.get("S09_SKIP_REPRO") == "1":
        return [check("reproducibility: rerun skipped (S09_SKIP_REPRO=1)", False, "not run", level=WARN)], {}
    tmp = out / "repro_processed"
    stages, seconds = [s for s in STAGES if s != STAGE], {}
    started = time.time()
    for stage in stages:
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "pipeline.run", "--data-dir", str(paths.root),
                               "--processed-dir", str(tmp), "--stage", stage], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2])
        seconds[stage] = round(time.time() - t0)
        if proc.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return [check("reproducibility: every stage reruns", False, f"{stage}: {proc.stdout[-300:]}"
                          f"{proc.stderr[-300:]}", level=WARN)], {"stage_seconds": seconds}
    total = round(time.time() - started)
    files = compare_finals(paths.processed / "final", tmp / "final")
    shutil.rmtree(tmp, ignore_errors=True)
    differ = [k for k, v in files.items() if not v["same"]]
    res = {"total_seconds": total, "stage_seconds": seconds, "files_compared": len(files), "differ": differ,
           "files": files}
    return [check("reproducibility: a full rerun gives identical final files", not differ,
                  f"{len(files) - len(differ)} of {len(files)} identical; full run {total // 60} min {total % 60} s",
                  level=WARN)], res


# ---------- App memory, upload ----------

def check_memory(final: Path) -> tuple[list[Check], dict]:
    path = final / "factor_panel_monthly.parquet"
    full = pd.read_parquet(path)
    mb_full = full.memory_usage(deep=True).sum() / 1e6
    del full
    few = pd.read_parquet(path, columns=APP_COLUMNS)
    mb_few = few.memory_usage(deep=True).sum() / 1e6
    cat = few.astype({"ticker": "category", "sector_name": "category"})
    mb_cat = cat.memory_usage(deep=True).sum() / 1e6
    res = {"full_mb": round(mb_full, 1), "columns_full": len(dictionary.columns("factor_panel_monthly.parquet")),
           "subset_columns": APP_COLUMNS,
           "subset_mb": round(mb_few, 1), "subset_categorical_mb": round(mb_cat, 1), "app_limit_mb": APP_MEMORY_MB}
    return [check(f"app memory: full panel at most half of the ~{APP_MEMORY_MB} MB free tier",
                  mb_full <= APP_MEMORY_MB / 2, f"full {mb_full:,.0f} MB; {len(APP_COLUMNS)} columns {mb_few:,.0f} MB "
                  f"({mb_cat:,.0f} MB with categorical ticker and sector)", level=WARN)], res


NO_NETWORK = """import runpy, socket, sys
def _blocked(*a, **k):
    raise OSError("network disabled for the offline dry run")
socket.socket.connect = _blocked
socket.create_connection = _blocked
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def check_upload(final: Path, out: Path, manifest: dict) -> tuple[list[Check], dict]:
    md = (final / "UPLOAD.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", md, re.S).group(1)
    script = out / "upload_wrds.py"
    script.write_text(code, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    proc = subprocess.run([sys.executable, "-c", NO_NETWORK, str(script), str(final), "--offline"],
                          capture_output=True, text=True, env=env, cwd=out)
    keys = re.findall(r"would upload s3://[^/]+/(\S+)", proc.stdout)
    prefix = manifest["s3_target"].split("/", 3)[3]
    n_files = sum(1 for f in final.rglob("*") if f.is_file())
    bad = [k for k in keys if not k.startswith(prefix)]
    res = {"exit_code": proc.returncode, "keys": len(keys), "files_in_final": n_files, "prefix": prefix,
           "outside_prefix": bad, "first_keys": keys[:3], "last_key": keys[-1] if keys else None,
           "stderr": proc.stderr[-300:]}
    planned = sum(o["files"] for o in manifest["outputs"].values()) + 4    # + daily _manifest.json and 3 documents
    res["planned_files"] = planned
    ok = proc.returncode == 0 and len(keys) == n_files == planned and not bad
    return [check(f"upload dry run offline: one key per planned file ({planned}) under {prefix}", ok,
                  f"{len(keys)} keys, {n_files} files, exit code {proc.returncode}", level=WARN)], res


# ---------- VALIDATION.md ----------

def verdict_line(checks: list[Check], prefix: str) -> str:
    mine = [c for c in checks if c.name.startswith(prefix)]
    return "**Verdict:** " + "; ".join(f"{c.status} - {c.name.split(': ', 1)[-1]} ({c.detail})" for c in mine)

def _pct(v, signed=False):
    return "n/a" if v is None else (f"{v:+.2%}" if signed else f"{v:.2%}")


def render(r: dict, checks: list[Check], manifest: dict) -> str:
    verdict = {c.name: c.status for c in checks}
    v = lambda prefix: " ".join(s for n, s in verdict.items() if n.startswith(prefix)) or "n/a"
    md = [f"# Validation of the WRDS data (version {manifest['data_version']})", "",
          f"Stage `s09_validate` of the pipeline (commit `{manifest['pipeline_git_commit'][:12]}`) checks the files "
          "in `processed/final/` against sources it did not build: the IWM ETF and the Russell 2000 index, the "
          "Fama-French factors, the old notebook's panel, and the raw CSVs read by a separate script. Every "
          "threshold is a warning (WARN), not a failure.", "",
          "| check | verdict |", "|---|---|"]
    for title, prefix in (("1. Index tracking", "index tracking"), ("2. Factor sanity", "factor sanity"),
                          ("3. Old vs new backtest", "old vs new"), ("4. Independent recompute", "independent"),
                          ("5. Reproducibility", "reproducibility"), ("6. App memory", "app memory"),
                          ("7. Upload dry run", "upload")):
        md.append(f"| {title} | {v(prefix)} |")
    t = r["tracking"]
    a, b, c = t["vs_iwm"], t["vs_r2000_price_index"], t["iwm_weighted_vs_iwm"]
    md += ["", "## 1. Index tracking", "",
           "Each month t: the panel's stocks weighted by `mkt_cap_musd` at t, with their `returns_monthly` return "
           "in month t+1, against IWM's total return (CRSP) and the Russell 2000 price index (no dividends). The "
           "same stocks weighted by their IWM fund weight (`index_weight`) show what the weighting alone does.", "",
           "| | correlation | tracking error a year | mean difference a year |", "|---|---|---|---|",
           f"| market-cap weighted vs IWM | {a['correlation']:.4f} | {_pct(a['tracking_error_annual'])} | "
           f"{_pct(a['mean_annual_difference'], True)} |",
           f"| market-cap weighted vs Russell 2000 price index | {b['correlation']:.4f} | "
           f"{_pct(b['tracking_error_annual'])} | {_pct(b['mean_annual_difference'], True)} |",
           f"| IWM-weight weighted vs IWM | {c['correlation']:.4f} | {_pct(c['tracking_error_annual'])} | "
           f"{_pct(c['mean_annual_difference'], True)} |", "",
           "| year | portfolio | IWM | difference (pp) | IWM-weighted difference (pp) | R2000 price | vs R2000 (pp) |",
           "|---|---|---|---|---|---|---|"]
    for y, row in a["by_year"].items():
        cw, rb = c["by_year"].get(y, {}), b["by_year"].get(y, {})
        md.append(f"| {y} | {_pct(row['portfolio'])} | {_pct(row['benchmark'])} | {row['difference_pp']:+.2f} | "
                  f"{cw.get('difference_pp', float('nan')):+.2f} | {_pct(rb.get('benchmark'))} | "
                  f"{rb.get('difference_pp', float('nan')):+.2f} |")
    if t["years_off_by_more_than_2pp"]:
        md += ["", "Years more than 2 pp from IWM, and the stocks that most separate market-cap weights from IWM's "
               "weights that year (contribution in pp):", ""]
        for y, e in t["years_off_by_more_than_2pp"].items():
            top = ", ".join(f"{x['ticker']} {x['contribution_pp']:+.2f}" for x in e["top_contributors_cap_vs_iwm_weight"])
            cause = ("most of it is market-cap vs IWM weighting"
                     if abs(e["iwm_weighted_difference_pp"]) < abs(e["difference_pp"]) / 2
                     else "the weighting is not the main cause")
            md.append(f"- **{y}** ({e['months']} months): {e['difference_pp']:+.2f} pp; with IWM's own weights "
                      f"{e['iwm_weighted_difference_pp']:+.2f} pp, so {cause}. Of the monthly differences, "
                      f"{e['july_august_pp']:+.2f} pp fall in July-August, when the snapshot in "
                      f"use (median {e['snapshot_age_days_median']:.0f} days old) still predates Russell's June "
                      f"reconstitution, and {e['other_months_pp']:+.2f} pp in the other months. The panel covers "
                      f"{e['iwm_weight_covered_pct']:.1f}% of IWM's weight on average. Largest weighting effects: "
                      f"{top}.")
    md += ["", verdict_line(checks, "index tracking"), ""]
    md += ["## 2. Factor sanity against Fama-French", "",
           "Each month t: within the panel, value-weighted (mkt_cap_musd) top 30% minus bottom 30% by the factor, "
           "return in month t+1, against the published Fama-French factor of that month (built on all US stocks, so "
           "a perfect match is not expected).", "",
           "| factor | Fama-French | correlation | long-short mean a month | t-stat | Fama-French mean a month |",
           "|---|---|---|---|---|---|"]
    for name, f in r["factors"].items():
        md.append(f"| {name} | {f['fama_french'].upper()} | {f['correlation']:.3f} | {_pct(f['mean_monthly'], True)} "
                  f"| {f['t_stat']:.2f} | {_pct(f['fama_french_mean_monthly'], True)} |")
    md += ["", "RMW sorts on operating profitability over book equity; gross profit over assets is a different "
           "measure, so ROE is shown next to it for reference.", "", verdict_line(checks, "factor sanity"), ""]
    bt = r.get("backtest", {})
    md += ["## 3. Old vs new backtest", ""]
    if bt:
        md += [f"Top 10% by ROA at each June month end, equal weight, held July to June, on this panel and on the "
               f"old notebook's panel (`{bt['old_panel']}`). {bt['method']}. Mean overlap of the June portfolios "
               f"(Jaccard): {bt['mean_june_overlap_jaccard']:.2f}.", "",
               "| | new | old | new, full holding | IWM |", "|---|---|---|---|---|",
               "| annualized | " + " | ".join(_pct(bt["annualized"][k]) for k in ("new", "old", "new_full_holding",
                                                                                  "iwm")) + " |", "",
               "| year | new | old | new - old (pp) | new, full holding | IWM |", "|---|---|---|---|---|---|"]
        for y, row in bt["by_year"].items():
            md.append(f"| {y} | {_pct(row['new'])} | {_pct(row['old'])} | {row['new_minus_old_pp']:+.2f} | "
                      f"{_pct(row['new_full_holding'])} | {_pct(row['iwm'])} |")
        if bt["years_more_than_3pp_apart"]:
            md += ["", "Years more than 3 pp apart, with how the June portfolios behind them differ:", ""]
            for y, e in bt["years_more_than_3pp_apart"].items():
                md.append(f"- **{y}:** " + "; ".join(f"{k}: {v['both']} in both, {v['new_only']} only new, "
                                                    f"{v['old_only']} only old" for k, v in e.items()))
    md.append("")
    rc = r.get("recompute", {})
    md += [verdict_line(checks, "old vs new"), "", "## 4. Independent recompute", ""]
    if rc:
        md += [f"{rc['rows']} random panel rows (seed {SEED}) recomputed from the raw CSVs by "
               "`scripts/wrds_independent_recompute.py`, which imports no pipeline code and re-implements the rules "
               f"from PLAN.md. A value matches when both are missing or they agree within {REL_TOL:g} relative.", "",
               "| value | compared | match | both missing |", "|---|---|---|---|"]
        for col, x in rc["metrics"].items():
            md.append(f"| {col} | {x['compared']} | {x['match']} | {x['both_missing']} |")
        if rc["mismatches"]:
            md += ["", "| permno | month | ticker | value | panel | raw | reason |", "|---|---|---|---|---|---|---|"]
            for x in rc["mismatches"]:
                md.append(f"| {x['permno']} | {x['date']} | {x['ticker']} | {x['metric']} | {x['panel']} | "
                          f"{x['raw']} | {x['reason']} |")
    md.append("")
    rp = r.get("reproducibility", {})
    md += [verdict_line(checks, "independent"), "", "## 5. Reproducibility", ""]
    if rp.get("files"):
        md += [f"Every stage s00-s08 rerun from raw/ into a temporary folder: {rp['total_seconds'] // 60} min "
               f"{rp['total_seconds'] % 60} s. {rp['files_compared'] - len(rp['differ'])} of {rp['files_compared']} "
               "final files identical (Parquet: rows and a hash of the data sorted by the key; JSON and Markdown: "
               "text, created_at ignored)." + (f" Different: {', '.join(rp['differ'])}." if rp["differ"] else ""), "",
               "Seconds per stage: " + ", ".join(f"{k} {v}" for k, v in rp["stage_seconds"].items()) + "."]
    else:
        md.append("Not run.")
    mem = r["memory"]
    md += ["", verdict_line(checks, "reproducibility"), "", "## 6. App memory", "",
           f"factor_panel_monthly loaded fully ({mem['columns_full']} columns): {mem['full_mb']:,.0f} MB in pandas. "
           f"With only {', '.join(mem['subset_columns'])}: {mem['subset_mb']:,.0f} MB, or "
           f"{mem['subset_categorical_mb']:,.0f} MB with ticker and sector_name as categoricals. Streamlit's free "
           f"tier has about {mem['app_limit_mb']:,} MB for the whole app.", ""]
    up = r["upload"]
    md += [verdict_line(checks, "app memory"), "", "## 7. Upload dry run", "",
           f"The script in UPLOAD.md run with `--offline`, without AWS variables and with network connections "
           f"blocked: exit code {up['exit_code']}, {up['keys']} S3 keys for {up['files_in_final']} files, all under "
           f"`{up['prefix']}`" + (f" except {up['outside_prefix']}" if up["outside_prefix"] else "") + ".", "",
           verdict_line(checks, "upload"), ""]
    return "\n".join(md)


# ---------- Stage ----------

def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, paths.stage_dir(STAGE).name)
    final, raw = paths.processed / "final", paths.raw
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(final / "factor_panel_monthly.parquet",
                            columns=["date", "permno", "ticker", "mkt_cap_musd", "index_weight", "snapshot_report_dt",
                                     "ret_next_1m",
                                     *{c for _, c, *_ in (*FACTOR_TESTS, *INFO_FACTOR_TESTS)}, "f_roa"])
    rets = pd.read_parquet(final / "returns_monthly.parquet", columns=["month_end", "permno", "ret_m"])
    checks, r = [], {}
    for key, fn in (("tracking", lambda: check_tracking(panel, rets, raw)),
                    ("factors", lambda: check_factors(panel, rets, raw)),
                    ("backtest", lambda: check_backtest(paths, panel, rets, raw)),
                    ("recompute", lambda: check_recompute(paths, out, final)),
                    ("memory", lambda: check_memory(final)),
                    ("upload", lambda: check_upload(final, out, manifest)),
                    ("reproducibility", lambda: check_reproducible(paths, out))):
        t0 = time.time()
        c, r[key] = fn()
        checks += c
        print(f"  {key}: {' '.join(x.status for x in c)} ({time.time() - t0:.0f}s)", flush=True)
    (out / "VALIDATION.md").write_text(render(r, checks, manifest), encoding="utf-8")
    summary = write_summary(out, STAGE, {}, checks, {"data_version": manifest["data_version"],
                                                     "pipeline_git_commit": manifest["pipeline_git_commit"],
                                                     **r})
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
