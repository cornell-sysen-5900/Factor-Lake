"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s05_ibes.py
PURPOSE: Stage s05_ibes: the full IBES consensus (7 measures, FPI 0-3, USD) linked to permno, with the CRSP
         factors that put IBES per-share values on the same share basis as a CRSP price.

In:  raw/forward_estimates.csv (1.1 GB, needed columns only), raw/crsp_ibes_link.csv,
     processed/s01_membership/snapshots.parquet (our permnos), processed/s02_prices/daily/ (date, permno, prc, cfacpr),
     raw/query_inputs/ibes_tickers.txt (only to compare)
Out: processed/s05_ibes/estimates_consensus.parquet  one row per permno, statpers, measure, fpi
     processed/s05_ibes/link.parquet                 the links used
     processed/s05_ibes/ibes_tickers.txt             the upload list for the IBES query
     processed/s05_ibes/summary.json

Rules (pipeline/PLAN.md, s05_ibes):
    1. CURCODE = USD; FPI 0-3 (FY4 dropped, decision 1)
    2. two different actuals for one (ticker, measure, fpedats): the actual is not used
    3. duplicate (ticker, statpers, measure, fpi): the row with more analysts, then the first in the file
    4. link: scores 1-2, our permnos, sdate <= statpers <= edate; each permno's latest link is open-ended
    5. FPI 0 (long-term growth) from percent to decimal; totals (SAL, EBT, EBI, CPX) are $ millions; a total more
       than 100x both its previous and next estimate (same ticker, measure, FPI; one neighbour at the ends) is NaN
       (suspect_total). Done on IBES tickers before linking, so share classes do not neighbour each other.
    6. share basis: IBES per-share values are restated to the latest share basis; CRSP prices are not.
       value x crsp_basis_factor (= cfacpr_statpers / cfacpr_last) is on the basis of the CRSP price at statpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from pipeline.checks import PASS, WARN, Check, check, vs_last_run
from pipeline.config import (IBES_CURRENCY, IBES_FPI, IBES_LINK_SCORES, IBES_MEASURES, IBES_OPEN_LINK_END,
                             IBES_PER_SHARE, IBES_TOTAL_OUTLIER_MULTIPLE, IBES_TOTALS, PRICE_ASOF_DAYS, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages.s00_inventory import read_header

STAGE = "s05_ibes"
RAW_FILE = "forward_estimates.csv"
LINK_FILE = "crsp_ibes_link.csv"
OUTPUT = "estimates_consensus.parquet"
VALUES = ("meanest", "medest", "stdev", "highest", "lowest")
_TYPES = {"ticker": pa.string(), "statpers": pa.date32(), "measure": pa.string(), "fpi": pa.string(),
          "curcode": pa.string(), "fpedats": pa.date32(), "anndats_act": pa.date32(),
          **{c: pa.float64() for c in ("numest", "numup", "numdown", "actual", *VALUES)}}
OUT_COLS = ("permno", "ibes_ticker", "statpers", "measure", "fpi", "fpedats", "numest", "numup", "numdown",
            *VALUES, "actual", "anndats_act", "price_statpers", "cfacpr_statpers", "cfacpr_last",
            "crsp_basis_factor", "suspect_total")


def read_estimates(path) -> pd.DataFrame:
    """Only the needed columns, typed; tickers stay text (`NA` is a real IBES ticker)."""
    original = {c.strip().lower(): c for c in read_header(path)}
    tbl = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=[original[c] for c in _TYPES], column_types={original[c]: t for c, t in _TYPES.items()}))
    ib = tbl.rename_columns([c.strip().lower() for c in tbl.column_names]).to_pandas(date_as_object=False)
    for c in ("statpers", "fpedats", "anndats_act"):
        ib[c] = ib[c].astype("datetime64[ns]")
    ib["ticker"] = ib["ticker"].str.strip()
    ib["fpi"] = pd.to_numeric(ib["fpi"], errors="coerce")
    ib["row"] = np.arange(len(ib))                                   # file order, only to break ties
    return ib


def filter_and_dedupe(ib: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rules 1-3."""
    n0 = len(ib)
    non_usd = ib["curcode"] != IBES_CURRENCY
    bad_fpi = ~ib["fpi"].isin([int(k) for k in IBES_FPI])
    ib = ib[~non_usd & ~bad_fpi].copy()
    ib["fpi"] = ib["fpi"].astype("int8")
    conflict = ib.groupby(["ticker", "measure", "fpedats"])["actual"].transform("nunique") > 1
    ib.loc[conflict, "actual"] = np.nan
    key = ["ticker", "statpers", "measure", "fpi"]
    n1 = len(ib)
    ib = (ib.sort_values(key + ["numest", "row"], ascending=[True] * 4 + [False, True])
            .drop_duplicates(key, keep="first"))
    return ib, {"raw_rows": n0, "non_usd_dropped": int(non_usd.sum()), "fpi_outside_0_3_dropped": int(bad_fpi.sum()),
                "conflicting_actuals_blanked": int(conflict.sum()), "duplicates_dropped": n1 - len(ib)}


def clean_links(lk: pd.DataFrame, ours: set) -> pd.DataFrame:
    """Rule 4: scores 1-2, our permnos; each permno's latest link (latest sdate) is open-ended."""
    lk = lk.copy()
    lk.columns = lk.columns.str.lower()
    lk = lk[lk["score"].isin(IBES_LINK_SCORES)].dropna(subset=["permno", "ticker"])
    lk["permno"] = lk["permno"].astype("int32")
    lk = lk[lk["permno"].isin(ours)].copy()
    lk["ticker"] = lk["ticker"].str.strip()
    latest = lk.groupby("permno")["sdate"].transform("max") == lk["sdate"]
    lk["edate_used"] = lk["edate"].where(~latest, pd.Timestamp(IBES_OPEN_LINK_END))
    return lk.sort_values(["permno", "sdate"]).reset_index(drop=True)


def link(ib: pd.DataFrame, lk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach permno when sdate <= statpers <= edate_used. If two tickers give the same permno, statpers, measure
    and fpi, keep the better link (score 1 before 2), then the one with more analysts."""
    m = ib.merge(lk[["ticker", "permno", "sdate", "edate_used", "score"]], on="ticker", how="inner")
    m = m[(m["statpers"] >= m["sdate"]) & (m["statpers"] <= m["edate_used"])]
    key = ["permno", "statpers", "measure", "fpi"]
    n = len(m)
    m = m.sort_values(key + ["score", "numest", "row"], ascending=[True] * 4 + [True, False, True])
    m = m.drop_duplicates(key, keep="first")
    return m, n - len(m)


def neighbour_outliers(ib: pd.DataFrame) -> pd.DataFrame:
    """The flagged total-measure estimates, with their previous and next estimate: more than 100x both neighbours of
    the same ticker, measure and FPI (only the one neighbour for the first and last). Neighbours that are missing or
    0 are skipped; with none left the estimate is not flagged."""
    tot = ib[ib["measure"].isin(IBES_TOTALS) & (ib["fpi"] > 0)].sort_values(["ticker", "measure", "fpi", "statpers"])
    g = tot.groupby(["ticker", "measure", "fpi"])["meanest"]
    tot = tot.assign(previous=g.shift(1), next=g.shift(-1))
    x, prev, nxt = tot["meanest"].abs(), tot["previous"].abs(), tot["next"].abs()
    prev, nxt = prev.where(prev > 0), nxt.where(nxt > 0)
    m = IBES_TOTAL_OUTLIER_MULTIPLE
    hit = (prev.notna() | nxt.notna()) & ((x > m * prev) | prev.isna()) & ((x > m * nxt) | nxt.isna())
    return tot[hit.to_numpy(dtype=bool)]


def convert_units(ib: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """Rule 5: long-term growth to decimals; totals far above both neighbouring estimates are NaN. Also returns the
    counts by measure and the 20 largest flagged estimates."""
    ltg = ib["fpi"] == 0
    ib.loc[ltg, list(VALUES)] = ib.loc[ltg, list(VALUES)] / 100
    flagged = neighbour_outliers(ib)
    sample = [{"ticker": r.ticker, "statpers": f"{r.statpers:%Y-%m-%d}", "measure": r.measure, "fpi": int(r.fpi),
               "meanest": float(r.meanest), "previous": None if pd.isna(r.previous) else float(r.previous),
               "next": None if pd.isna(r.next) else float(r.next)}
              for r in flagged.sort_values("meanest", key=abs, ascending=False).head(20).itertuples()]
    suspect = ib.index.isin(flagged.index)
    ib.loc[suspect, list(VALUES)] = np.nan
    ib["suspect_total"] = suspect
    return ib, flagged.groupby("measure").size().to_dict(), sample


def add_crsp_basis(ib: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Rule 6: CRSP price and cfacpr on statpers (last trading day with a price, at most 7 days earlier), and the
    permno's last cfacpr. crsp_basis_factor = cfacpr_statpers / cfacpr_last."""
    px = daily.dropna(subset=["prc", "cfacpr"])
    px = px[px["cfacpr"] > 0].sort_values(["permno", "date"])
    last = px.groupby("permno")["cfacpr"].last().rename("cfacpr_last")
    px = px.rename(columns={"prc": "price_statpers", "cfacpr": "cfacpr_statpers"}).sort_values("date")
    ib = pd.merge_asof(ib.sort_values("statpers"), px[["date", "permno", "price_statpers", "cfacpr_statpers"]],
                       left_on="statpers", right_on="date", by="permno", direction="backward",
                       tolerance=pd.Timedelta(days=PRICE_ASOF_DAYS)).drop(columns="date")
    ib["cfacpr_last"] = ib["permno"].map(last)
    ib["crsp_basis_factor"] = ib["cfacpr_statpers"] / ib["cfacpr_last"]
    return ib


def run_checks(out: pd.DataFrame, lk: pd.DataFrame, info: dict, raw_list: set | None, tickers: set) -> list[Check]:
    checks = [check("only USD and FPI 0-3", bool(out["fpi"].isin([0, 1, 2, 3]).all())),
              check("unique permno, statpers, measure, fpi",
                    not out.duplicated(["permno", "statpers", "measure", "fpi"]).any()),
              check("every measure present", set(IBES_MEASURES) <= set(out["measure"]),
                    ", ".join(sorted(set(IBES_MEASURES) - set(out["measure"]))) or "all 7")]
    w = out.merge(lk[["ticker", "permno", "sdate", "edate_used"]].rename(columns={"ticker": "ibes_ticker"}),
                  on=["ibes_ticker", "permno"], how="left")
    inside = (w["statpers"] >= w["sdate"]) & (w["statpers"] <= w["edate_used"])
    checks.append(check("every row is inside one of its links' date windows",
                        bool(inside.groupby([w["permno"], w["statpers"], w["measure"], w["fpi"]]).any().all())))
    ltg = out.loc[(out["measure"] == "EPS") & (out["fpi"] == 0), "meanest"].median()
    checks.append(check("long-term growth in decimals (EPS FPI 0 median between 0.03 and 0.30)",
                        bool(0.03 <= ltg <= 0.30), f"median {ltg:.3f}"))
    tot = out["measure"].isin(IBES_TOTALS) & (out["fpi"] > 0)
    share = float(out.loc[tot, "suspect_total"].mean())
    checks.append(check("suspect totals (100x both neighbouring estimates) under 0.01% of total-measure rows",
                        share < 1e-4, f"{int(out['suspect_total'].sum())} rows ({share:.4%})", level=WARN))
    eps1 = out[(out["measure"] == "EPS") & (out["fpi"] == 1)]
    found = float(eps1["price_statpers"].notna().mean())
    checks.append(check("CRSP price on statpers for at least 90% of EPS FY1 rows", found >= 0.90, f"{found:.1%}",
                        level=WARN))
    restated = float((eps1["crsp_basis_factor"].dropna() != 1).mean())
    checks.append(check("crsp_basis_factor > 0 wherever present", bool((out["crsp_basis_factor"].dropna() > 0).all()),
                        f"{restated:.1%} of EPS FY1 rows have a split between statpers and the end of the data"))
    checks += [vs_last_run("EPS FY1 linked rows", len(eps1), "ibes_eps_fy1_rows"),
               vs_last_run("EPS FY1 permnos", int(eps1["permno"].nunique()), "ibes_eps_fy1_permnos")]
    if raw_list is not None:
        checks.append(check("ibes_tickers.txt equals raw/query_inputs/ibes_tickers.txt", tickers == raw_list,
                            f"{len(tickers - raw_list)} new, {len(raw_list - tickers)} dropped (if not 0: rerun the "
                            "IBES query)", level=WARN))
    return checks


def run(paths: DataPaths) -> dict:
    out_dir = fresh_stage_dir(paths, STAGE)
    ours = set(pd.read_parquet(paths.stage_dir("s01_membership") / "snapshots.parquet", columns=["permno"])
               ["permno"].unique().tolist())
    lk = clean_links(pd.read_csv(paths.raw / LINK_FILE, dtype={"TICKER": str, "ncusip": str}, keep_default_na=False,
                                 na_values=[""], parse_dates=["sdate", "edate"]), ours)
    write_parquet(lk, out_dir / "link.parquet")
    tickers = set(lk["ticker"])
    (out_dir / "ibes_tickers.txt").write_text("\n".join(lk["ticker"].drop_duplicates()) + "\n")

    ib, info = filter_and_dedupe(read_estimates(paths.raw / RAW_FILE))
    ib, suspects, suspect_sample = convert_units(ib)
    ib, info["two_tickers_dropped"] = link(ib, lk)
    daily = pd.read_parquet(paths.stage_dir("s02_prices") / "daily", columns=["date", "permno", "prc", "cfacpr"])
    ib = add_crsp_basis(ib, daily)
    del daily
    out = (ib.rename(columns={"ticker": "ibes_ticker"})[list(OUT_COLS)]
             .sort_values(["permno", "statpers", "measure", "fpi"]).reset_index(drop=True))
    raw_path = paths.query_inputs / "ibes_tickers.txt"
    raw_list = set(x.strip() for x in raw_path.read_text().splitlines() if x.strip()) if raw_path.exists() else None
    checks = run_checks(out, lk, info, raw_list, tickers)
    write_parquet(out, out_dir / OUTPUT)

    year = out["statpers"].dt.year
    eps1 = (out["measure"] == "EPS") & (out["fpi"] == 1)
    by_year = pd.DataFrame({"rows": out.groupby(year).size(), "permnos": out.groupby(year)["permno"].nunique(),
                            "eps_fy1_permnos": out[eps1].groupby(year[eps1])["permno"].nunique()})
    by_mf = out.groupby(["measure", "fpi"])
    extra = {**info, "suspect_totals": suspects, "suspect_totals_sample": suspect_sample, "linked_rows": len(out),
             "linked_permnos": int(out["permno"].nunique()),
             "links_used": len(lk), "link_permnos": int(lk["permno"].nunique()), "our_permnos": len(ours),
             "rows_by_measure_fpi": {f"{m} {f}": int(n) for (m, f), n in by_mf.size().items()},
             "coverage_by_measure_fpi": {f"{m} {f}": round(float(v), 4) for (m, f), v in
                                         by_mf["meanest"].apply(lambda s: s.notna().mean()).items()},
             "by_year": {str(y): {k: int(v) for k, v in r.items()} for y, r in by_year.fillna(0).iterrows()}}
    stats = {f"{m}_fpi{f}_meanest": out.loc[(out["measure"] == m) & (out["fpi"] == f), "meanest"]
             for m in IBES_MEASURES for f in (0, 1, 2, 3)}
    tables = {OUTPUT: table_info(out, "statpers", ["numest", "meanest", "stdev", "price_statpers",
                                                    "crsp_basis_factor"]),
              "meanest by measure and fpi": {k: table_info(pd.DataFrame({"v": v}), None, ["v"])["stats"]["v"]
                                             for k, v in stats.items()},
              "link.parquet": table_info(lk, "sdate", [])}
    summary = write_summary(out_dir, STAGE, tables, checks, extra)
    print(f"  {info['raw_rows']:,} raw rows -> {len(out):,} linked rows, {out['permno'].nunique():,} permnos | "
          f"non-USD {info['non_usd_dropped']:,}, duplicates {info['duplicates_dropped']:,}, "
          f"two tickers {info['two_tickers_dropped']}, suspect totals {suspects}")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
