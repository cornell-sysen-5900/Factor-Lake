"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s01_membership.py
PURPOSE: Stage s01_membership: IWM snapshots, membership periods, permno fill from CUSIP, known gaps, and the
         non-member (merger) stocks excluded from the universe.

In:  raw/iwm_holdings_raw.csv, raw/r2000_price_crsp_ciz.csv (permno, date, flag, price, shares: month-end market
     caps for rule 6, computed by s02's own function), raw/query_inputs/permno_list.txt (only to compare)
Out: processed/s01_membership/snapshots.parquet             one row per report_dt and permno
     processed/s01_membership/periods.parquet               one row per membership period
     processed/s01_membership/excluded_non_members.parquet  stock-snapshots excluded from the universe
     processed/s01_membership/known_gaps.parquet, known_gaps.md
     processed/s01_membership/permno_list.txt               the upload list for the permno-based WRDS queries
     processed/s01_membership/summary.json

Rules (pipeline/PLAN.md, s01_membership):
    1. a line without a permno gets the most common permno of the same CUSIP on other dates (audit A8)
    2. lines still without a permno that look like common stock are the known gaps (A8)
    3. one row per stock per snapshot: weights summed, earliest eff_dt, names from the largest line
    4. a stock missing from 2+ consecutive snapshots starts a new period; one missing snapshot is bridged (A7)
    5. permno_list.txt = every snapshot permno
    6. a stock-snapshot is out of the universe if (a) its month-end market cap is > 3x the largest other stock in
       the snapshot, or (b) the stock is in <= 3 snapshots in total, all with weight 0.00% (decision 11)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check, vs_last_run
from pipeline.config import (KNOWN_MISSING_MONTHS, MEMBERSHIP_BRIDGE_SNAPSHOTS, NEIGHBOUR_MISSING_MAX,
                             NON_MEMBER_CAP_MULTIPLE, NON_MEMBER_MAX_SNAPSHOTS, NON_STOCK_PATTERN,
                             SNAPSHOT_SIZE_RANGE, SNAPSHOT_WEIGHT_RANGE, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages import s02_prices

STAGE = "s01_membership"
HOLDINGS_FILE = "iwm_holdings_raw.csv"


def load_holdings(path) -> pd.DataFrame:
    h = pd.read_csv(path, dtype={"cusip": str, "ticker": str, "security_name": str}, keep_default_na=False,
                    na_values=[""], parse_dates=["report_dt", "eff_dt"])
    h.columns = h.columns.str.lower()
    h["line"] = np.arange(len(h))                      # file order, only to break ties
    return h


def fill_permno_from_cusip(h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rule 1: a line without a permno gets the most common permno of the same CUSIP on other lines."""
    known = h.dropna(subset=["permno", "cusip"])
    per_cusip = known.groupby("cusip")["permno"].nunique()
    mapping = known.groupby("cusip")["permno"].agg(lambda s: s.mode().iloc[0])
    fill = h["permno"].isna() & h["cusip"].map(mapping).notna()
    h = h.copy()
    h.loc[fill, "permno"] = h.loc[fill, "cusip"].map(mapping)
    h["permno_from_cusip"] = fill
    return h, {"lines": int(fill.sum()), "securities": int(h.loc[fill, "cusip"].nunique()),
               "cusips_with_two_permnos": sorted(per_cusip[per_cusip > 1].index)}


def known_gaps(h: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: lines still without a permno that have a CUSIP and a ticker and do not look like cash, futures,
    rights and the like. One row per security."""
    name = h["security_name"].fillna("").str.upper()
    gap = h[h["permno"].isna() & h["cusip"].notna() & h["ticker"].notna()
            & ~name.str.contains(NON_STOCK_PATTERN, regex=True)]
    out = (gap.sort_values(["report_dt", "line"]).groupby("cusip")
              .agg(security_name=("security_name", "last"), ticker=("ticker", "last"), first=("report_dt", "min"),
                   last=("report_dt", "max"), snapshots=("report_dt", "nunique"), max_weight=("percent_tna", "max"))
              .reset_index())
    return out.sort_values(["snapshots", "cusip"], ascending=[False, True]).reset_index(drop=True)


def build_snapshots(h: pd.DataFrame) -> pd.DataFrame:
    """Rule 3: one row per (report_dt, permno). Names and CUSIP come from the line with the largest weight, then
    the most shares (equal weights happen when a SPAC and its new name are both listed)."""
    h = h.dropna(subset=["permno"]).assign(permno=lambda d: d["permno"].astype("int32"))
    keys = ["report_dt", "permno"]
    agg = h.groupby(keys).agg(eff_dt=("eff_dt", "min"), weight=("percent_tna", "sum"),
                              n_lines=("percent_tna", "size"), permno_from_cusip=("permno_from_cusip", "any"))
    names = (h.sort_values(keys + ["percent_tna", "nbr_shares", "line"], ascending=[True, True, False, False, True])
              .groupby(keys)[["ticker", "security_name", "cusip"]].first())
    snap = agg.join(names).reset_index()
    return snap[["report_dt", "eff_dt", "permno", "weight", "ticker", "security_name", "cusip", "n_lines",
                 "permno_from_cusip"]].sort_values(keys).reset_index(drop=True)


def apply_non_member_rules(snap: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    """Rule 6: adds mkt_cap_musd (month-end of report_dt), max_other_mkt_cap_musd, n_snapshots_total,
    exclusion_rule ('', 'a', 'b', 'a+b') and in_universe."""
    snap = snap.assign(month=snap["report_dt"].dt.to_period("M"))
    snap = snap.merge(caps[["permno", "month", "mkt_cap_musd"]], on=["permno", "month"], how="left")
    mc = snap["mkt_cap_musd"]
    by = snap.groupby("report_dt")["mkt_cap_musd"]
    max1 = by.transform("max")
    max2 = by.transform(lambda s: s.nlargest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
    is_top = mc.eq(max1)
    unique_top = is_top & is_top.groupby(snap["report_dt"]).transform("sum").eq(1)
    snap["max_other_mkt_cap_musd"] = np.where(unique_top, max2, max1)
    rule_a = (mc > NON_MEMBER_CAP_MULTIPLE * snap["max_other_mkt_cap_musd"]).fillna(False)

    per = snap.groupby("permno").agg(n=("report_dt", "nunique"), w=("weight", lambda w: w.round(2).abs().max()))
    small = per.index[(per["n"] <= NON_MEMBER_MAX_SNAPSHOTS) & (per["w"] == 0)]
    rule_b = snap["permno"].isin(small)

    snap["n_snapshots_total"] = snap["permno"].map(per["n"]).astype("int32")
    snap["exclusion_rule"] = np.select([rule_a & rule_b, rule_a, rule_b], ["a+b", "a", "b"], default="")
    snap["in_universe"] = snap["exclusion_rule"].eq("")
    return snap.drop(columns="month")


def build_periods(snap: pd.DataFrame) -> pd.DataFrame:
    """Rule 4: membership periods over all snapshots; excluded_non_member when every snapshot of the period is
    excluded from the universe."""
    order = {d: i for i, d in enumerate(sorted(snap["report_dt"].unique()))}
    s = snap.assign(k=snap["report_dt"].map(order)).sort_values(["permno", "k"])
    s["period"] = (s.groupby("permno")["k"].diff().fillna(np.inf) > 1 + MEMBERSHIP_BRIDGE_SNAPSHOTS).cumsum()
    s["excluded"] = ~s["in_universe"]
    p = s.groupby("period").agg(permno=("permno", "first"), index_effective=("report_dt", "min"),
                                index_through=("report_dt", "max"), first_known=("eff_dt", "min"),
                                n_snapshots=("report_dt", "size"), n_excluded=("excluded", "sum"),
                                ticker=("ticker", "last"), security_name=("security_name", "last"))
    p["excluded_non_member"] = p["n_excluded"].eq(p["n_snapshots"])
    cols = ["permno", "index_effective", "index_through", "first_known", "n_snapshots", "n_excluded",
            "excluded_non_member", "ticker", "security_name"]
    return p[cols].sort_values(["permno", "index_effective"]).reset_index(drop=True)


def gaps_markdown(gaps: pd.DataFrame, n_dates: int) -> str:
    md = ["# Known gaps: IWM stocks without a permno", "",
          "Written by pipeline stage s01_membership. These holdings look like common stock but have no permno on any "
          "IWM report date (not even under the same CUSIP), so they are not in the snapshots, prices or "
          "fundamentals. Decision (audit A8): no extra WRDS pull; keep them listed here.", "",
          f"{len(gaps)} securities, {n_dates} snapshot dates affected.", "",
          "| security | ticker | CUSIP | first | last | snapshots | max weight (%) |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r.security_name} | {r.ticker} | {r.cusip} | {r.first:%Y-%m-%d} | {r.last:%Y-%m-%d} | {r.snapshots} "
           f"| {r.max_weight:.2f} |" for r in gaps.itertuples()]
    return "\n".join(md) + "\n"


def run_checks(snap, periods, h, fill, listed, raw_list) -> list[Check]:
    checks = [check("unique report_dt and permno", not snap.duplicated(["report_dt", "permno"]).any()),
              check("eff_dt on or after report_dt", bool((snap["eff_dt"] >= snap["report_dt"]).all())),
              check("no CUSIP maps to two permnos", not fill["cusips_with_two_permnos"],
                    ", ".join(fill["cusips_with_two_permnos"][:10]), level=WARN)]
    lo, hi = SNAPSHOT_SIZE_RANGE
    size = snap[snap["in_universe"]].groupby("report_dt").size()
    bad = size[(size < lo) | (size > hi)]
    checks.append(check(f"in-universe stocks per snapshot within {lo:,}-{hi:,}", bad.empty,
                        f"{size.min():,} to {size.max():,}" + (f"; outside: {bad.to_dict()}" if len(bad) else ""),
                        level=WARN))
    lo, hi = SNAPSHOT_WEIGHT_RANGE
    w = snap.groupby("report_dt")["weight"].sum()
    bad = w[(w < lo) | (w > hi)]
    checks.append(check(f"weight sum per snapshot within {lo:.0f}-{hi:.0f}%", bad.empty,
                        f"{w.min():.1f}% to {w.max():.1f}%" + (f"; outside: {bad.round(1).to_dict()}"
                                                               if len(bad) else ""), level=WARN))
    dates = pd.Series(sorted(snap["report_dt"].unique()))
    missing = [str(m) for m in pd.period_range(dates.min(), dates.max(), freq="M").difference(dates.dt.to_period("M"))]
    checks.append(check("missing months are the known ones", missing == list(KNOWN_MISSING_MONTHS),
                        ", ".join(missing) or "none", level=WARN))
    sets = snap.groupby("report_dt")["permno"].apply(set)
    gap = [len((sets.iloc[i - 1] & sets.iloc[i + 1]) - sets.iloc[i]) for i in range(1, len(sets) - 1)]
    checks.append(check(f"stocks missing from a snapshot but in both neighbours at most {NEIGHBOUR_MISSING_MAX}",
                        max(gap, default=0) <= NEIGHBOUR_MISSING_MAX, f"max {max(gap, default=0)}", level=WARN))
    if raw_list is not None:
        checks.append(check("permno_list.txt equals raw/query_inputs/permno_list.txt", listed == raw_list,
                            f"{len(listed - raw_list)} new, {len(raw_list - listed)} dropped "
                            "(if not 0: rerun the permno-based WRDS queries)", level=WARN))
    no_cap = snap["mkt_cap_musd"].isna().mean()
    checks.append(check("snapshot stocks with a CIZ month-end market cap (rule 6a)", no_cap <= 0.01,
                        f"{no_cap:.2%} without one", level=WARN))
    checks += [vs_last_run("snapshots", snap["report_dt"].nunique(), "snapshots"),
               vs_last_run("snapshot rows", len(snap), "snapshot_rows"),
               vs_last_run("permnos", snap["permno"].nunique(), "permnos"),
               vs_last_run("periods", len(periods), "periods"),
               vs_last_run("lines filled from CUSIP", fill["lines"], "cusip_fill_lines")]
    return checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    h, fill = fill_permno_from_cusip(load_holdings(paths.raw / HOLDINGS_FILE))
    gaps = known_gaps(h)
    snap = build_snapshots(h)

    px = s02_prices.read_ciz(paths.raw / s02_prices.CIZ_FILE, s02_prices.MARKET_CAP_COLUMNS)
    px, _, _ = s02_prices.drop_duplicates(px)
    caps = s02_prices.month_end_market_caps(s02_prices.clean_prices(px))
    del px
    snap = apply_non_member_rules(snap, caps)
    periods = build_periods(snap)
    excluded = snap.loc[~snap["in_universe"], ["report_dt", "eff_dt", "permno", "ticker", "security_name",
                                                "mkt_cap_musd", "max_other_mkt_cap_musd", "weight",
                                                "n_snapshots_total", "exclusion_rule"]].rename(
        columns={"exclusion_rule": "rule"}).reset_index(drop=True)

    listed = set(snap["permno"].tolist())
    raw_path = paths.query_inputs / "permno_list.txt"
    raw_list = set(int(x) for x in raw_path.read_text().split()) if raw_path.exists() else None
    checks = run_checks(snap, periods, h, fill, listed, raw_list)

    write_parquet(snap, out / "snapshots.parquet")
    write_parquet(periods, out / "periods.parquet")
    write_parquet(excluded, out / "excluded_non_members.parquet")
    write_parquet(gaps, out / "known_gaps.parquet")
    n_gap_dates = int(h.loc[h["permno"].isna() & h["cusip"].isin(gaps["cusip"]), "report_dt"].nunique())
    (out / "known_gaps.md").write_text(gaps_markdown(gaps, n_gap_dates), encoding="utf-8")
    (out / "permno_list.txt").write_text("\n".join(str(p) for p in sorted(listed)) + "\n")

    tables = {"snapshots.parquet": table_info(snap, "report_dt", ["weight", "mkt_cap_musd", "n_lines"]),
              "periods.parquet": table_info(periods, "index_effective", ["n_snapshots"]),
              "excluded_non_members.parquet": table_info(excluded, "report_dt", ["mkt_cap_musd", "weight"]),
              "known_gaps.parquet": table_info(gaps, "first", ["snapshots", "max_weight"])}
    extra = {"in_universe_rows": int(snap["in_universe"].sum()),
             "cusip_fill": {k: v for k, v in fill.items() if k != "cusips_with_two_permnos"},
             "excluded_non_members": [
                 {"security_name": r.security_name, "ticker": r.ticker, "permno": int(r.permno),
                  "report_dt": f"{r.report_dt:%Y-%m-%d}", "eff_dt": f"{r.eff_dt:%Y-%m-%d}",
                  "mkt_cap_musd": None if pd.isna(r.mkt_cap_musd) else round(float(r.mkt_cap_musd), 1),
                  "max_other_mkt_cap_musd": None if pd.isna(r.max_other_mkt_cap_musd)
                  else round(float(r.max_other_mkt_cap_musd), 1),
                  "weight_pct": round(float(r.weight), 4), "n_snapshots_total": int(r.n_snapshots_total),
                  "rule": r.rule} for r in excluded.itertuples()]}
    summary = write_summary(out, STAGE, tables, checks, extra)
    print(f"  {snap['report_dt'].nunique()} snapshots, {len(snap):,} rows, {len(listed):,} permnos, "
          f"{len(periods):,} periods | {fill['lines']:,} lines filled from CUSIP | {len(gaps)} known gaps | "
          f"{len(excluded)} stock-snapshots excluded ({excluded['permno'].nunique()} stocks)")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
