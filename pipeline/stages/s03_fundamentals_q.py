"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s03_fundamentals_q.py
PURPOSE: Stage s03_fundamentals_q: quarterly Compustat statements -> 17 factors at quarter end plus the RV and
         DCF building blocks, one row per permno per fiscal quarter.

In:  raw/fundamentals_raw.csv (CCM quarterly, 67 columns)
Out: processed/s03_fundamentals_q/fundamentals_quarterly.parquet, summary.json

Rules (pipeline/PLAN.md, s03_fundamentals_q; ported from the notebook's FUNDAMENTALS cell):
    1. one row per (permno, fyearq, fqtr): the most complete row, then the latest rdq
    2. report_date = rdq, else datadate + 45 days (fiscal Q1-Q3) or + 90 days (Q4)
    3. year-to-date cash flows -> quarterly (Q1 = YTD, else YTD - previous quarter of the same fiscal year);
       TTM = exactly 4 consecutive fiscal quarters; growth vs exactly 4 quarters earlier
    4. per-share items split-adjusted with ajexq before summing, restated in this quarter's shares (audit D4)
    5. impossible values to NaN; non-USD rows and prices below $0.01 get no price-based ratios (decisions 2, 14)
    6. interest is never filled with 0 (decision 13); tax rate = taxes / pretax income when pretax > 0, 0-60%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check, vs_last_run
from pipeline.config import (COVERAGE_TOLERANCE, GICS_INDUSTRIES, GICS_SECTORS, LAST_RUN_COVERAGE_2015,
                             PRICE_BASED_FACTORS, PRICE_FLOOR, REPORT_LAG_DAYS_Q, REPORT_LAG_DAYS_Q4,
                             MIN_SUE_STD, MIN_TTM_CAPEX_MUSD, MIN_TTM_SALES_MUSD, SUE_HISTORY_QUARTERS,
                             SUE_MIN_HISTORY, TAX_RATE_RANGE, YTD_ITEMS, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary

STAGE = "s03_fundamentals_q"
RAW_FILE = "fundamentals_raw.csv"
OUTPUT = "fundamentals_quarterly.parquet"
_TEXT = ("gvkey", "tic", "conm", "indfmt", "consol", "popsrc", "datafmt", "curcdq", "datacqtr", "datafqtr",
         "costat")
FACTORS = ("f_book_to_price", "f_earnings_to_price", "f_sales_to_price", "f_ebitda_to_ev", "f_dividend_yield",
           "f_roe", "f_roa", "f_gross_profitability", "f_accruals_to_assets", "f_leverage", "f_current_ratio",
           "f_rd_to_sales", "f_asset_growth_1yr", "f_sales_growth_1yr", "f_capex_growth_1yr", "f_net_share_issuance",
           "f_sue")
# TTM sums: output column -> quarterly column
TTM_ITEMS = {"revenue_ttm": "saleq", "revt_ttm": "revtq", "cogs_ttm": "cogsq", "xsga_ttm": "xsgaq",
             "xrd_ttm": "xrdq", "ebitda_ttm": "oibdpq", "ebit_ttm": "oiadpq", "da_ttm": "dpq",
             "interest_ttm": "xintq", "pretax_ttm": "piq", "tax_ttm": "txtq", "ni_ttm": "niq", "ib_ttm": "ibq",
             "oancf_ttm": "oancf_q", "capx_ttm": "capx_q", "dv_ttm": "dv_q", "sstk_ttm": "sstk_q",
             "prstkc_ttm": "prstkc_q", "dltis_ttm": "dltis_q", "dltr_ttm": "dltr_q", "aqc_ttm": "aqc_q"}
BLOCKS = ("book_equity", "total_debt", "cash", "net_debt", "preferred", "minority_interest", "ev_claims", "nwc",
          "d_nwc_4q", *TTM_ITEMS, "fcf_ttm", "tax_rate_ttm", "epsfxq_ttm_adj", "dvpsxq_ttm_adj", "diluted_shares",
          "csho_adj", "cshfd_adj", "mkt_cap_qe_musd")
ID_COLS = ("permno", "gvkey", "tic", "conm", "period_end", "report_date", "report_date_source", "rdq", "fyearq",
           "fqtr", "curcdq", "is_usd", "ajexq", "gsector", "gsector_name", "ggroup", "gind", "gind_name", "gsubind",
           "naics", "price_qe", "cshoq")
REPORTED = ("atq", "ltq", "actq", "lctq", "cheq", "invtq", "rectq", "apq", "ppentq", "gdwlq", "intanq", "dlcq",
            "dlttq", "pstkq", "mibtq", "seqq", "ceqq", "txditcq", "saleq", "revtq", "cogsq", "xsgaq", "xrdq",
            "oibdpq", "dpq", "oiadpq", "xintq", "piq", "txtq", "niq", "ibq", "epsfxq", "epspxq", "dvpsxq",
            *YTD_ITEMS.values())


def load_quarterly(path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    text = {c: str for c in head if c.lower() in _TEXT}
    fq = pd.read_csv(path, dtype=text, keep_default_na=False, na_values=[""], low_memory=False)
    fq.columns = fq.columns.str.lower()
    fq = fq.rename(columns={"lpermno": "permno"})
    for c in ("datadate", "rdq"):
        fq[c] = pd.to_datetime(fq[c], errors="coerce")
    return fq


def dedupe_quarters(fq: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rule 1: one row per (permno, fiscal quarter); the most complete row wins, then the latest rdq."""
    fq = fq.dropna(subset=["permno", "datadate", "fyearq", "fqtr"]).copy()
    fq["permno"] = fq["permno"].astype("int32")
    fq["fyearq"] = fq["fyearq"].astype(int)
    fq["fqtr"] = fq["fqtr"].astype(int)
    fq["q"] = fq["fyearq"] * 4 + fq["fqtr"]
    fq["filled"] = fq.notna().sum(axis=1)
    n = len(fq)
    fq = (fq.sort_values(["permno", "q", "filled", "rdq"], na_position="first")
            .drop_duplicates(["permno", "q"], keep="last").drop(columns="filled")
            .sort_values(["permno", "q"]).reset_index(drop=True))
    return fq, n - len(fq)


def add_report_date(fq: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: rdq, else datadate + 45 days (fiscal Q1-Q3) or + 90 days (Q4)."""
    lag = np.where(fq["fqtr"] == 4, REPORT_LAG_DAYS_Q4, REPORT_LAG_DAYS_Q)
    fq["report_date"] = fq["rdq"].fillna(fq["datadate"] + pd.to_timedelta(lag, unit="D"))
    fq["report_date_source"] = np.where(fq["rdq"].notna(), "rdq", "proxy")
    return fq


class Quarters:
    """Values from exactly k fiscal quarters earlier for the same permno (NaN if that quarter is missing)."""

    def __init__(self, fq: pd.DataFrame):
        self.fq = fq
        self.index = pd.MultiIndex.from_arrays([fq["permno"], fq["q"]])

    def lag(self, values, k: int) -> np.ndarray:
        s = pd.Series(np.asarray(values), index=self.index)
        return s.reindex(pd.MultiIndex.from_arrays([self.fq["permno"], self.fq["q"] - k])).to_numpy()

    def ttm(self, values) -> np.ndarray:
        """Sum of this and the 3 previous fiscal quarters; NaN if any of the 4 is missing."""
        v = np.asarray(values, dtype="float64")
        return v + self.lag(v, 1) + self.lag(v, 2) + self.lag(v, 3)


def ytd_to_quarterly(fq: pd.DataFrame, qs: Quarters) -> pd.DataFrame:
    """Rule 3: Q1 = YTD; later quarters = YTD - previous quarter's YTD of the same fiscal year, else NaN."""
    same_year = qs.lag(fq["fyearq"], 1) == fq["fyearq"].to_numpy()
    for raw, out in YTD_ITEMS.items():
        fq[out] = np.where(fq["fqtr"] == 1, fq[raw], np.where(same_year, fq[raw] - qs.lag(fq[raw], 1), np.nan))
    return fq


def add_building_blocks(fq: pd.DataFrame, qs: Quarters) -> pd.DataFrame:
    for out, col in TTM_ITEMS.items():
        fq[out] = qs.ttm(fq[col])
    # per-share items split-adjusted before summing, then restated in this quarter's shares (audit D4)
    fq["epsfxq_ttm_adj"] = qs.ttm(fq["epsfxq"] / fq["ajexq"]) * fq["ajexq"]
    fq["dvpsxq_ttm_adj"] = qs.ttm(fq["dvpsxq"] / fq["ajexq"]) * fq["ajexq"]
    pref = fq["pstkq"].fillna(0)
    book = fq["seqq"].fillna(fq["ceqq"] + pref).fillna(fq["atq"] - fq["ltq"])
    fq["book_equity"] = book + fq["txditcq"].fillna(0) - pref               # Fama-French style
    fq["total_debt"] = fq["dlttq"].fillna(0) + fq["dlcq"].fillna(0)
    fq["cash"] = fq["cheq"]
    fq["net_debt"] = fq["total_debt"] - fq["cheq"].fillna(0)
    fq["preferred"] = fq["pstkq"]
    fq["minority_interest"] = fq["mibtq"]
    fq["ev_claims"] = fq["net_debt"] + pref + fq["mibtq"].fillna(0)
    fq["nwc"] = (fq["actq"] - fq["cheq"].fillna(0)) - (fq["lctq"] - fq["dlcq"].fillna(0))
    fq["d_nwc_4q"] = fq["nwc"] - qs.lag(fq["nwc"], 4)
    fq["fcf_ttm"] = fq["oancf_ttm"] - fq["capx_ttm"]
    lo, hi = TAX_RATE_RANGE
    fq["tax_rate_ttm"] = (fq["tax_ttm"] / fq["pretax_ttm"]).where(fq["pretax_ttm"] > 0).clip(lo, hi)
    fq["diluted_shares"] = fq["cshfdq"]
    fq["csho_adj"] = fq["cshoq"] * fq["ajexq"]
    fq["cshfd_adj"] = fq["cshfdq"] * fq["ajexq"]
    fq["price_qe"] = fq["prccq"]
    fq["mkt_cap_qe_musd"] = fq["prccq"] * fq["cshoq"]
    return fq


def add_factors(fq: pd.DataFrame, qs: Quarters) -> tuple[pd.DataFrame, dict]:
    """The notebook's 17 factors at quarter end, then rule 5."""
    be = fq["book_equity"].where(fq["book_equity"] > 0)
    mcap = fq["mkt_cap_qe_musd"]
    ev = (mcap + fq["net_debt"]).where(lambda x: x > 0)
    f = pd.DataFrame(index=fq.index)
    f["f_book_to_price"] = be / mcap
    f["f_earnings_to_price"] = fq["epsfxq_ttm_adj"] / fq["prccq"]
    f["f_sales_to_price"] = fq["revenue_ttm"] / mcap
    f["f_ebitda_to_ev"] = fq["ebitda_ttm"] / ev
    f["f_dividend_yield"] = fq["dvpsxq_ttm_adj"] / fq["prccq"]
    f["f_roe"] = fq["ni_ttm"] / be
    f["f_roa"] = fq["ni_ttm"] / fq["atq"]
    f["f_gross_profitability"] = (fq["revt_ttm"] - fq["cogs_ttm"]) / fq["atq"]
    f["f_accruals_to_assets"] = (fq["ni_ttm"] - fq["oancf_ttm"]) / fq["atq"]
    f["f_leverage"] = fq["total_debt"] / fq["atq"]
    f["f_current_ratio"] = fq["actq"] / fq["lctq"].where(fq["lctq"] > 0)
    # near-zero denominators give meaningless ratios: TTM sales and capex need $1M, the SUE std $0.01
    f["f_rd_to_sales"] = fq["xrd_ttm"].fillna(0) / fq["revenue_ttm"].where(fq["revenue_ttm"] >= MIN_TTM_SALES_MUSD)
    atq4, sale4, capx4 = qs.lag(fq["atq"], 4), qs.lag(fq["revenue_ttm"], 4), qs.lag(fq["capx_ttm"], 4)
    f["f_asset_growth_1yr"] = fq["atq"] / atq4 - 1
    f["f_sales_growth_1yr"] = fq["revenue_ttm"] / np.where(sale4 >= MIN_TTM_SALES_MUSD, sale4, np.nan) - 1
    f["f_capex_growth_1yr"] = fq["capx_ttm"] / np.where(capx4 >= MIN_TTM_CAPEX_MUSD, capx4, np.nan) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        f["f_net_share_issuance"] = np.log(fq["csho_adj"] / qs.lag(fq["csho_adj"], 4))
    # SUE: change in split-adjusted EPS vs 4 quarters earlier / std of the previous 8 such changes
    eps = fq["epspxq"] / fq["ajexq"]
    d_eps = eps - qs.lag(eps, 4)
    hist = pd.DataFrame({k: qs.lag(d_eps, k) for k in range(1, SUE_HISTORY_QUARTERS + 1)}, index=fq.index)
    std = hist.std(axis=1)
    f["f_sue"] = d_eps / std.where((hist.notna().sum(axis=1) >= SUE_MIN_HISTORY) & (std >= MIN_SUE_STD))
    f = f.replace([np.inf, -np.inf], np.nan)

    bad = {"f_sales_growth_1yr": f["f_sales_growth_1yr"] < -1, "f_capex_growth_1yr": f["f_capex_growth_1yr"] < -1,
           "f_rd_to_sales": f["f_rd_to_sales"] < 0}
    for col, mask in bad.items():
        f.loc[mask, col] = np.nan
    no_price = ~fq["is_usd"] | (fq["prccq"] < PRICE_FLOOR)
    f.loc[no_price, list(PRICE_BASED_FACTORS)] = np.nan
    fq.loc[~fq["is_usd"], "mkt_cap_qe_musd"] = np.nan
    counts = {"impossible_to_nan": {k: int(v.sum()) for k, v in bad.items()},
              "non_usd_rows": int((~fq["is_usd"]).sum()), "non_usd_permnos": int(fq.loc[~fq["is_usd"], "permno"].nunique()),
              "price_below_floor_rows": int((fq["prccq"] < PRICE_FLOOR).sum())}
    return pd.concat([fq, f], axis=1), counts


def add_gics(fq: pd.DataFrame) -> pd.DataFrame:
    for c in ("gsector", "ggroup", "gind", "gsubind", "naics"):
        fq[c] = pd.to_numeric(fq[c], errors="coerce").astype("Int64")
    fq["gsector_name"] = fq["gsector"].map(GICS_SECTORS).astype(object)
    fq["gind_name"] = fq["gind"].map(GICS_INDUSTRIES).astype(object)
    return fq


def build(fq_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fq, n_dups = dedupe_quarters(fq_raw)
    fq = add_report_date(fq)
    fq["is_usd"] = fq["curcdq"].eq("USD")
    qs = Quarters(fq)
    fq = add_building_blocks(ytd_to_quarterly(fq, qs), qs)
    fq, counts = add_factors(fq, qs)
    fq = add_gics(fq).rename(columns={"datadate": "period_end"})
    fq = fq[[*ID_COLS, *REPORTED, *BLOCKS, *FACTORS]]
    return fq, {"duplicates_removed": n_dups, **counts}


def run_checks(fq: pd.DataFrame, info: dict, seed: int = 7) -> list[Check]:
    checks = [check("unique permno and fiscal quarter", not fq.duplicated(["permno", "fyearq", "fqtr"]).any()),
              vs_last_run("duplicate quarters removed", info["duplicates_removed"], "fundamentals_duplicates"),
              vs_last_run("rows", len(fq), "fundamentals_rows")]
    early = int((fq["report_date"] < fq["period_end"]).sum())
    checks.append(check("report_date on or after period_end", early == 0, f"{early} rows earlier", level=WARN))
    proxy = float((fq["report_date_source"] == "proxy").mean())
    checks.append(check("report_date from the rdq proxy at most 2%", proxy <= 0.02, f"{proxy:.2%}", level=WARN))
    no_price = ~fq["is_usd"] | (fq["price_qe"] < PRICE_FLOOR)
    leaked = int(fq.loc[no_price, list(PRICE_BASED_FACTORS)].notna().sum().sum())
    checks.append(check("no price-based ratio on non-USD rows or prices below $0.01", leaked == 0,
                        f"{info['non_usd_rows']} non-USD rows ({info['non_usd_permnos']} permnos), "
                        f"{info['price_below_floor_rows']} rows below ${PRICE_FLOOR}"))
    checks.append(check("ajexq > 0", bool((fq["ajexq"] > 0).all()), f"{int((~(fq['ajexq'] > 0)).sum())} rows not"))
    lo, hi = TAX_RATE_RANGE
    t = fq["tax_rate_ttm"].dropna()
    checks.append(check(f"tax rate within {lo:.0%}-{hi:.0%}", bool(t.between(lo, hi).all()), f"{len(t):,} rows"))
    # interest is never filled: a row whose 4 quarters miss any xintq has no interest_ttm
    qs = Quarters(fq.assign(q=fq["fyearq"] * 4 + fq["fqtr"]))
    complete = np.isfinite(qs.ttm(fq["xintq"]))
    checks.append(check("interest never filled (interest_ttm only from 4 reported xintq)",
                        bool((fq["interest_ttm"].notna() == complete).all()),
                        f"{int(fq['interest_ttm'].notna().sum()):,} rows with interest_ttm"))

    # TTM against an independent sum for 20 random rows that have 3 earlier quarters
    rng = np.random.default_rng(seed)
    rows = fq.loc[rng.choice(fq.index, min(200, len(fq)), replace=False)]
    keyed = fq.set_index(["permno", "fyearq", "fqtr"])["saleq"]
    def prev(y, q, k):
        q0 = y * 4 + q - 1 - k
        return q0 // 4, q0 % 4 + 1
    hand = []
    for r in rows.itertuples():
        parts = [keyed.get((r.permno, *prev(r.fyearq, r.fqtr, k)), np.nan) for k in range(4)]
        hand.append((r.revenue_ttm, float(np.sum(parts)) if not np.isnan(parts).any() else np.nan))
    hand = np.array(hand).T
    checks.append(check("revenue_ttm equals the sum of 4 fiscal quarters (200 random rows)",
                        bool(np.allclose(hand[0], hand[1], equal_nan=True)),
                        f"{int(np.isfinite(hand[1]).sum())} rows with 4 quarters"))

    recent = fq[fq["period_end"] >= "2015-01-01"]
    cov = {c: float(recent[c].notna().mean()) for c in FACTORS}
    off = {c: f"{cov[c]:.0%} (last run {LAST_RUN_COVERAGE_2015[c]:.0%})" for c in FACTORS
           if abs(cov[c] - LAST_RUN_COVERAGE_2015[c]) > COVERAGE_TOLERANCE}
    checks.append(check(f"factor coverage 2015+ within {COVERAGE_TOLERANCE:.0%} of the last run", not off,
                        "; ".join(f"{k}: {v}" for k, v in off.items()) or "all 17 within", level=WARN))
    unnamed = sorted(int(g) for g in fq.loc[fq["gind"].notna() & fq["gind_name"].isna(), "gind"].unique())
    checks.append(check("every GICS industry code has a name", not unnamed,
                        f"no name for {unnamed} ({int(fq['gind'].isin(unnamed).sum())} rows)", level=WARN))
    return checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    raw = load_quarterly(paths.raw / RAW_FILE)
    fq, info = build(raw)
    checks = run_checks(fq, info)
    write_parquet(fq, out / OUTPUT)

    recent = fq[fq["period_end"] >= "2015-01-01"]
    odd = {"saleq_negative": int((fq["saleq"] < 0).sum()), "dpq_negative": int((fq["dpq"] < 0).sum()),
           "cshoq_zero": int((fq["cshoq"] == 0).sum()), "atq_zero": int((fq["atq"] == 0).sum())}
    extra = {**info, "report_date_source": fq["report_date_source"].value_counts().to_dict(), "odd_values": odd,
             "coverage_2015_onward": {c: round(float(recent[c].notna().mean()), 4) for c in FACTORS},
             "coverage_all": {c: round(float(fq[c].notna().mean()), 4) for c in (*FACTORS, *BLOCKS)}}
    tables = {OUTPUT: table_info(fq, "period_end", [*FACTORS, *BLOCKS])}
    summary = write_summary(out, STAGE, tables, checks, extra)
    print(f"  {len(raw):,} raw rows -> {len(fq):,} quarters ({info['duplicates_removed']} duplicates), "
          f"{fq['permno'].nunique():,} permnos | proxy report dates {extra['report_date_source'].get('proxy', 0)} | "
          f"non-USD {info['non_usd_rows']} rows | impossible to NaN {info['impossible_to_nan']}")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
