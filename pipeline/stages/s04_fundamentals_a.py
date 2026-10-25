"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s04_fundamentals_a.py
PURPOSE: Stage s04_fundamentals_a: annual Compustat statements -> DCF inputs, one row per permno per fiscal year.

In:  raw/fundamentals_annual_raw.csv (CCM annual, 68 columns),
     processed/s03_fundamentals_q/fundamentals_quarterly.parquet (rdq of the Q4 rows, TTM revenue to cross-check)
Out: processed/s04_fundamentals_a/dcf_inputs_annual.parquet, summary.json

Rules (pipeline/PLAN.md, s04_fundamentals_a):
    1. one row per (permno, fyear): the most complete row, then the latest datadate
    2. report_date = rdq of the quarterly Q4 row with the same gvkey and fiscal year, else datadate + 90 days
    3. changes and growth only vs exactly fyear - 1
    4. interest expense = xint, else xintd; never filled with 0 (decision 13)
    5. tax_rate = txt / pi when pi > 0, clipped to 0-60% (the reported effective rate); tax_rate_fcff = tax_rate,
       or 0% when pi <= 0 (no tax on a loss); FCFF = EBIT x (1 - tax_rate_fcff) + D&A - capex - change in NWC,
       missing if any part is missing; statutory_tax_rate 35% up to fiscal 2017, 21% from 2018
    6. dcf_applicable = False for GICS sectors 40 (Financials) and 60 (Real Estate)
    7. own table: not merged into the monthly panel (decision 6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check
from pipeline.config import (DCF_EXCLUDED_SECTORS, GICS_INDUSTRIES, GICS_SECTORS, REPORT_LAG_DAYS_ANNUAL,
                             STATUTORY_TAX_RATE, STATUTORY_TAX_RATE_FROM_2018, TAX_RATE_RANGE, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages.s03_fundamentals_q import OUTPUT as QUARTERLY_OUTPUT
from pipeline.stages.s03_fundamentals_q import STAGE as QUARTERLY_STAGE

STAGE = "s04_fundamentals_a"
RAW_FILE = "fundamentals_annual_raw.csv"
OUTPUT = "dcf_inputs_annual.parquet"
_TEXT = ("gvkey", "tic", "conm", "indfmt", "consol", "popsrc", "datafmt", "curcd", "costat")
FCFF_PARTS = ("ebit", "tax_rate_fcff", "da", "capex", "d_nwc")
BLOCKS = ("revenue", "ebitda", "ebit", "da", "capex", "oancf", "pretax_income", "taxes", "tax_rate",
          "tax_rate_fcff", "statutory_tax_rate", "interest_expense", "net_income", "nwc", "d_nwc", "total_debt", "cash", "net_debt", "preferred",
          "minority_interest", "diluted_shares", "basic_shares", "book_equity", "dividends", "dividends_total",
          "buybacks", "share_issuance", "fcff", "revenue_growth", "ebitda_growth")
ID_COLS = ("permno", "gvkey", "tic", "conm", "fyear", "fyr", "period_end", "report_date", "report_date_source",
           "curcd", "is_usd", "ajex", "price_fye", "epsfx", "emp", "sich", "gsector", "gsector_name", "ggroup",
           "gind", "gind_name", "gsubind", "naics", "dcf_applicable", "interest_source")


def load_annual(path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    fa = pd.read_csv(path, dtype={c: str for c in head if c.lower() in _TEXT}, keep_default_na=False,
                     na_values=[""], low_memory=False)
    fa.columns = fa.columns.str.lower()
    fa = fa.rename(columns={"lpermno": "permno"})
    fa["datadate"] = pd.to_datetime(fa["datadate"], errors="coerce")
    return fa


def dedupe_years(fa: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rule 1: one row per (permno, fyear); the most complete row wins, then the latest datadate."""
    fa = fa.dropna(subset=["permno", "datadate", "fyear"]).copy()
    fa["permno"] = fa["permno"].astype("int32")
    fa["fyear"] = fa["fyear"].astype(int)
    fa["filled"] = fa.notna().sum(axis=1)
    n = len(fa)
    fa = (fa.sort_values(["permno", "fyear", "filled", "datadate"])
            .drop_duplicates(["permno", "fyear"], keep="last").drop(columns="filled")
            .sort_values(["permno", "fyear"]).reset_index(drop=True))
    return fa, n - len(fa)


def add_report_date(fa: pd.DataFrame, quarterly: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rule 2: the rdq of the Q4 row with the same gvkey and fiscal year that ends on the same date; otherwise
    datadate + 90 days. After a fiscal-year-end change the Q4 of the same fiscal year can be an earlier quarter
    (e.g. June instead of December), and its rdq would predate the annual period. Also returns how many such Q4
    rows were rejected."""
    q4 = (quarterly.loc[(quarterly["fqtr"] == 4) & quarterly["rdq"].notna(), ["gvkey", "fyearq", "rdq", "period_end"]]
          .sort_values("rdq").drop_duplicates(["gvkey", "fyearq"], keep="first")
          .rename(columns={"fyearq": "fyear", "period_end": "q4_period_end"}))
    fa = fa.merge(q4, on=["gvkey", "fyear"], how="left")
    rejected = fa["rdq"].notna() & (fa["q4_period_end"] != fa["datadate"])
    fa.loc[rejected, "rdq"] = pd.NaT
    fa["report_date"] = fa["rdq"].fillna(fa["datadate"] + pd.Timedelta(days=REPORT_LAG_DAYS_ANNUAL))
    fa["report_date_source"] = np.where(fa["rdq"].notna(), "q4_rdq", "proxy")
    return fa.drop(columns=["rdq", "q4_period_end"]), int(rejected.sum())


def previous_year(fa: pd.DataFrame, values) -> np.ndarray:
    """Rule 3: the value of exactly fiscal year - 1 for the same permno (NaN if that year is missing)."""
    s = pd.Series(np.asarray(values, dtype="float64"), index=pd.MultiIndex.from_arrays([fa["permno"], fa["fyear"]]))
    return s.reindex(pd.MultiIndex.from_arrays([fa["permno"], fa["fyear"] - 1])).to_numpy()


def add_blocks(fa: pd.DataFrame) -> pd.DataFrame:
    fa["revenue"], fa["ebitda"], fa["ebit"], fa["da"] = fa["sale"], fa["oibdp"], fa["oiadp"], fa["dp"]
    fa["capex"], fa["oancf"], fa["net_income"] = fa["capx"], fa["oancf"], fa["ni"]
    fa["pretax_income"], fa["taxes"] = fa["pi"], fa["txt"]
    lo, hi = TAX_RATE_RANGE
    fa["tax_rate"] = (fa["txt"] / fa["pi"]).where(fa["pi"] > 0).clip(lo, hi)
    fa["tax_rate_fcff"] = fa["tax_rate"].where(~(fa["pi"] <= 0), 0.0)            # no tax on a loss
    fa["statutory_tax_rate"] = np.where(fa["fyear"] <= max(STATUTORY_TAX_RATE), STATUTORY_TAX_RATE[2017],
                                        STATUTORY_TAX_RATE_FROM_2018)
    fa["interest_expense"] = fa["xint"].fillna(fa["xintd"])                     # never 0-filled
    fa["interest_source"] = np.select([fa["xint"].notna(), fa["xintd"].notna()], ["xint", "xintd"], "missing")
    fa["total_debt"] = fa["dltt"].fillna(0) + fa["dlc"].fillna(0)
    fa["cash"] = fa["che"]
    fa["net_debt"] = fa["total_debt"] - fa["che"].fillna(0)
    fa["preferred"], fa["minority_interest"] = fa["pstk"], fa["mib"]
    fa["nwc"] = (fa["act"] - fa["che"].fillna(0)) - (fa["lct"] - fa["dlc"].fillna(0))
    fa["d_nwc"] = fa["nwc"] - previous_year(fa, fa["nwc"])
    fa["diluted_shares"], fa["basic_shares"] = fa["cshfd"], fa["csho"]
    pref = fa["pstk"].fillna(0)
    book = fa["seq"].fillna(fa["ceq"] + pref).fillna(fa["at"] - fa["lt"])
    fa["book_equity"] = book + fa["txditc"].fillna(0) - pref                    # Fama-French style, as s03
    fa["dividends"], fa["dividends_total"] = fa["dvc"], fa["dv"]
    fa["buybacks"], fa["share_issuance"] = fa["prstkc"], fa["sstk"]
    # FCFF; NaN arithmetic leaves it missing when any part is missing
    fa["fcff"] = fa["ebit"] * (1 - fa["tax_rate_fcff"]) + fa["da"] - fa["capex"] - fa["d_nwc"]
    prev_rev, prev_ebitda = previous_year(fa, fa["revenue"]), previous_year(fa, fa["ebitda"])
    fa["revenue_growth"] = fa["revenue"] / np.where(prev_rev > 0, prev_rev, np.nan) - 1
    fa["ebitda_growth"] = fa["ebitda"] / np.where(prev_ebitda > 0, prev_ebitda, np.nan) - 1
    fa[["revenue_growth", "ebitda_growth"]] = fa[["revenue_growth", "ebitda_growth"]].replace([np.inf, -np.inf], np.nan)
    fa["price_fye"] = fa["prcc_f"]
    fa["is_usd"] = fa["curcd"].eq("USD")
    for c in ("gsector", "ggroup", "gind", "gsubind", "naics", "sich"):
        fa[c] = pd.to_numeric(fa[c], errors="coerce").astype("Int64")
    fa["gsector_name"] = fa["gsector"].map(GICS_SECTORS).astype(object)
    fa["gind_name"] = fa["gind"].map(GICS_INDUSTRIES).astype(object)
    fa["dcf_applicable"] = ~fa["gsector"].isin(DCF_EXCLUDED_SECTORS).fillna(False).astype(bool)
    return fa


def build(fa_raw: pd.DataFrame, quarterly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fa, n_dups = dedupe_years(fa_raw)
    fa, differs = add_report_date(fa, quarterly)
    fa = add_blocks(fa).rename(columns={"datadate": "period_end"})
    return fa[[*ID_COLS, *BLOCKS]], {"duplicates_removed": n_dups, "q4_rejected_period_end_differs": differs}


def run_checks(fa: pd.DataFrame, quarterly: pd.DataFrame, info: dict) -> list[Check]:
    checks = [check("unique permno and fiscal year", not fa.duplicated(["permno", "fyear"]).any(),
                    f"{info['duplicates_removed']} duplicates removed")]
    early = int((fa["report_date"] < fa["period_end"]).sum())
    checks.append(check("report_date on or after period_end", early == 0, f"{early} rows earlier", level=WARN))
    proxy = float((fa["report_date_source"] == "proxy").mean())
    checks.append(check("report_date from the +90 day proxy at most 10%", proxy <= 0.10, f"{proxy:.2%}", level=WARN))
    lo, hi = TAX_RATE_RANGE
    t = fa["tax_rate"].dropna()
    checks.append(check(f"tax rate within {lo:.0%}-{hi:.0%}", bool(t.between(lo, hi).all()), f"{len(t):,} rows"))
    zero = int((fa["interest_source"].eq("missing") & fa["interest_expense"].notna()).sum())
    checks.append(check("interest never filled when xint and xintd are missing", zero == 0,
                        f"{int(fa['interest_source'].eq('missing').sum()):,} rows without interest"))
    parts = fa[list(FCFF_PARTS)].notna().all(axis=1)
    checks.append(check("FCFF present exactly when every part is present",
                        bool((fa["fcff"].notna() == parts).all()), f"{int(parts.sum()):,} rows with FCFF"))
    # annual revenue vs the quarterly TTM at the matching Q4 (same gvkey, fiscal year)
    q4 = (quarterly.loc[quarterly["fqtr"] == 4, ["gvkey", "fyearq", "revenue_ttm"]]
          .drop_duplicates(["gvkey", "fyearq"]).rename(columns={"fyearq": "fyear"}))
    m = fa[["gvkey", "fyear", "revenue"]].drop_duplicates(["gvkey", "fyear"]).merge(q4, on=["gvkey", "fyear"])
    m = m.dropna(subset=["revenue", "revenue_ttm"])
    m = m[m["revenue"].abs() > 1]                               # $1m floor: tiny revenues give noisy ratios
    close = float((np.abs(m["revenue_ttm"] / m["revenue"] - 1) <= 0.01).mean()) if len(m) else 1.0
    checks.append(check("annual revenue within 1% of the quarterly TTM for at least 90% of matched years "
                        "(informational: restatements change annual figures)", close >= 0.90,
                        f"{close:.1%} of {len(m):,}", level=WARN))
    return checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    quarterly = pd.read_parquet(paths.stage_dir(QUARTERLY_STAGE) / QUARTERLY_OUTPUT,
                                columns=["gvkey", "fyearq", "fqtr", "rdq", "period_end", "revenue_ttm"])
    fa, info = build(load_annual(paths.raw / RAW_FILE), quarterly)
    checks = run_checks(fa, quarterly, info)
    write_parquet(fa, out / OUTPUT)

    extra = {**info, "report_date_source": fa["report_date_source"].value_counts().to_dict(),
             "interest_source": fa["interest_source"].value_counts().to_dict(),
             "odd_values": {"capex_negative": int((fa["capex"] < 0).sum()),
                            "revenue_negative": int((fa["revenue"] < 0).sum())},
             "coverage": {c: round(float(fa[c].notna().mean()), 4) for c in BLOCKS},
             "dcf_applicable_rows": int(fa["dcf_applicable"].sum()),
             "fcff_coverage_dcf_applicable": round(float(fa.loc[fa["dcf_applicable"], "fcff"].notna().mean()), 4),
             "fcff_parts_missing_dcf_applicable": {c: round(float(fa.loc[fa["dcf_applicable"], c].isna().mean()), 4)
                                                  for c in FCFF_PARTS}}
    summary = write_summary(out, STAGE, {OUTPUT: table_info(fa, "period_end", list(BLOCKS))}, checks, extra)
    print(f"  {len(fa):,} fiscal years, {fa['permno'].nunique():,} permnos | report date: "
          f"{extra['report_date_source']} | interest: {extra['interest_source']} | "
          f"FCFF for {extra['coverage']['fcff']:.0%} of all, {extra['fcff_coverage_dcf_applicable']:.0%} of "
          f"{extra['dcf_applicable_rows']:,} DCF-applicable years")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
