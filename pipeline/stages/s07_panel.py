"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s07_panel.py
PURPOSE: Stage s07_panel: the monthly point-in-time factor panel, the RV multiples and peer statistics, and the
         security master.

In:  processed/s01_membership/snapshots.parquet, permno_list.txt; processed/s02_prices/monthly.parquet,
     calendar.parquet, daily/; processed/s03_fundamentals_q/fundamentals_quarterly.parquet;
     processed/s05_ibes/estimates_consensus.parquet, link.parquet; processed/s06_market/market_daily.parquet,
     market_monthly.parquet
Out: processed/s07_panel/factor_panel_monthly.parquet   one row per universe stock per month end t
     processed/s07_panel/rv_multiples_monthly.parquet   same rows: multiples and their industry/sector ranks
     processed/s07_panel/rv_peer_stats_monthly.parquet  one row per GICS industry per t
     processed/s07_panel/security_master.parquet        one row per permno in permno_list.txt
     processed/s07_panel/summary.json

Rules (pipeline/PLAN.md, s07_panel): everything at t uses only what was public by t: the latest snapshot whose
every line is effective (eff_dt <= t); prices from the last real trading day of month t; the latest fiscal quarter
with report_date <= t (a late-filed older quarter never replaces a newer one; stale after 365 days); the latest IBES
statpers <= t within 60 days; surprises announced on or before t within 120 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check
from pipeline.config import (BETA_DAYS, BETA_MONTHS, DISPERSION_MIN_MEAN, DISPERSION_MIN_NUMEST, GICS_INDUSTRIES,
                             GICS_SECTORS, IBES_MAX_AGE_DAYS, MIN_OBS_SHARE, PEER_MIN, PRICE_BASED_FACTORS,
                             PRICE_FLOOR, SANITY_MEDIAN_RANGES, SHARE_CLASS_GAP_COMPANY, SHARE_CLASS_GAP_YEAR,
                             SIC_TO_GICS_SECTOR, SIC_UNCLASSIFIED, SNAPSHOT_SIZE_RANGE, STALE_FUNDAMENTALS_DAYS,
                             SURPRISE_MAX_AGE_DAYS, SURPRISE_PRICE_MAX_DAYS, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages import s02_prices

STAGE = "s07_panel"
PANEL, RV, PEERS, MASTER = ("factor_panel_monthly.parquet", "rv_multiples_monthly.parquet",
                            "rv_peer_stats_monthly.parquet", "security_master.parquet")
STATEMENT_FACTORS = ("f_roe", "f_roa", "f_gross_profitability", "f_accruals_to_assets", "f_leverage",
                     "f_current_ratio", "f_rd_to_sales", "f_asset_growth_1yr", "f_sales_growth_1yr",
                     "f_capex_growth_1yr", "f_net_share_issuance", "f_sue")
QE_FACTORS = tuple(f"{c}_qe" for c in PRICE_BASED_FACTORS)
BLOCKS = ("book_equity", "ib_ttm", "revenue_ttm", "dv_ttm", "fcf_ttm", "ev_claims", "ebitda_ttm", "ebit_ttm", "cshoq")
PRICE_SCALED = ("f_book_to_price", "f_earnings_to_price", "f_sales_to_price", "f_dividend_yield", "f_fcf_yield",
                "f_ebitda_to_ev", "f_ebit_to_ev", "f_sales_to_ev")
ANALYST = ("f_forward_ep", "f_forward_ep_fy2", "f_forward_cfp", "f_forward_sales_to_ev", "f_forward_ebitda_to_ev",
           "f_ltg", "f_eps_revision", "f_eps_dispersion", "f_analyst_surprise")
PRICE_FACTORS = ("mom_1m", "mom_6m", "mom_12m", "mom_12_1", *s02_prices.VOL_COLS)
MULTIPLES = ("pe", "pb", "ps", "ev_ebitda", "ev_ebit", "ev_sales", "fcf_yield", "dividend_yield", "earnings_yield",
             "fwd_pe_fy1", "fwd_pe_fy2", "fwd_ev_ebitda", "fwd_ev_sales", "ltg", "peg")
SERIES = {"eps1": ("EPS", 1), "eps2": ("EPS", 2), "cps1": ("CPS", 1), "sal1": ("SAL", 1), "ebt1": ("EBT", 1),
          "ltg": ("EPS", 0)}


# ---------- Universe ----------

def rebalance_dates(calendar: pd.DataFrame, snaps: pd.DataFrame) -> pd.Series:
    """Month ends from the first month with an effective snapshot to the last month with prices."""
    first = snaps.groupby("report_dt")["eff_dt"].max().min()
    ends = calendar["month_end"].sort_values()
    return ends[ends >= first].reset_index(drop=True)


def build_universe(dates: pd.Series, snaps: pd.DataFrame) -> pd.DataFrame:
    """At each t: the latest snapshot whose every line is effective by t, in-universe lines only."""
    avail = snaps.groupby("report_dt")["eff_dt"].max()
    pick = pd.DataFrame({"date": dates.to_numpy(),
                         "snapshot_report_dt": [avail.index[avail <= t].max() for t in dates]})
    u = pick.merge(snaps[snaps["in_universe"]].rename(columns={"report_dt": "snapshot_report_dt"}),
                   on="snapshot_report_dt")
    u["snapshot_eff_dt"] = u["snapshot_report_dt"].map(avail)
    u = u.rename(columns={"weight": "index_weight", "ticker": "ticker_snapshot"})
    nxt = dict(zip(dates.iloc[:-1], dates.iloc[1:]))
    prev = u[["date", "permno"]].assign(date=u["date"].map(nxt), was_in=True).dropna(subset=["date"])
    u = u.merge(prev, on=["date", "permno"], how="left")
    u["new_entrant"] = u["was_in"].isna() & (u["date"] > dates.min())
    return u[["date", "permno", "snapshot_report_dt", "snapshot_eff_dt", "index_weight", "ticker_snapshot",
              "security_name", "new_entrant"]]


# ---------- Market data, betas ----------

def rolling_beta(y: pd.Series, x: pd.Series, g: pd.Series, n: int, min_obs: float) -> pd.Series:
    """cov(y, x) / var(x) over the last n rows of each group, where both are present; NaN below min_obs."""
    ok = (y.notna() & x.notna()).astype("float64")
    xs, ys = x.where(ok > 0, 0.0), y.where(ok > 0, 0.0)

    def last_n(v):
        c = v.groupby(g).cumsum()
        return c - c.groupby(g).shift(n).fillna(0)
    k, sx, sy, sxy, sxx = last_n(ok), last_n(xs), last_n(ys), last_n(xs * ys), last_n(xs * xs)
    var = sxx - sx * sx / k
    return ((sxy - sx * sy / k) / var).where((k >= min_obs) & (var > 1e-12))


def daily_month_end(daily: pd.DataFrame, mkt_d: pd.DataFrame) -> pd.DataFrame:
    """Per permno and month: cfacpr and beta_252d on the last real trading day of the month. daily is sorted."""
    m = mkt_d.set_index("date")
    x = daily["date"].map(m["mktrf"])
    y = daily["ret"] - daily["date"].map(m["rf"])
    beta = rolling_beta(y, x, daily["permno"], BETA_DAYS, MIN_OBS_SHARE * BETA_DAYS)
    rows = s02_prices.month_end_rows(daily)
    return pd.DataFrame({"permno": daily.loc[rows, "permno"].to_numpy(),
                         "month": daily.loc[rows, "date"].dt.to_period("M").to_numpy(),
                         "cfacpr_t": daily.loc[rows, "cfacpr"].to_numpy(), "beta_252d": beta[rows].to_numpy()})


def monthly_beta(monthly: pd.DataFrame, mkt_m: pd.DataFrame) -> pd.DataFrame:
    """beta_60m from monthly excess returns over the last 60 calendar months (at least 48)."""
    mo = monthly[["permno", "month_end", "ret_m"]].assign(month=monthly["month_end"].dt.to_period("M"))
    lo = mo.groupby("permno")["month"].min()
    hi = mo.groupby("permno")["month"].max()
    n = np.array([(h - l).n + 1 for l, h in zip(lo, hi)])
    grid = pd.DataFrame({"permno": np.repeat(lo.index.to_numpy(), n),
                         "month": np.concatenate([pd.period_range(l, h, freq="M") for l, h in zip(lo, hi)])})
    grid = grid.merge(mo[["permno", "month", "ret_m"]], on=["permno", "month"], how="left")
    mm = mkt_m.assign(month=pd.PeriodIndex(mkt_m["month"], freq="M")).set_index("month")
    x = grid["month"].map(mm["mktrf"])
    y = grid["ret_m"] - grid["month"].map(mm["rf"])
    grid["beta_60m"] = rolling_beta(y, x, grid["permno"], BETA_MONTHS, MIN_OBS_SHARE * BETA_MONTHS)
    return grid[["permno", "month", "beta_60m"]]


def add_market(u: pd.DataFrame, monthly: pd.DataFrame, me: pd.DataFrame, beta_m: pd.DataFrame) -> pd.DataFrame:
    mo = monthly.copy()
    # company market cap: every permno of ours with the same permco, at the same month end
    grp = mo.groupby(["month_end", "permco"])["mkt_cap_musd"]
    mo["company_mkt_cap_musd"] = grp.transform(lambda s: s.sum(min_count=1))
    mo["n_share_classes"] = grp.transform("count")
    cols = ["permno", "month_end", "permco", "ticker", "siccd", "last_trade_date", "prc", "shrout", "mkt_cap_musd",
            "company_mkt_cap_musd", "n_share_classes", "mom_1m", "mom_6m", "mom_12m", *s02_prices.VOL_COLS]
    u = u.merge(mo[cols].rename(columns={"month_end": "date", "last_trade_date": "price_date"}),
                on=["date", "permno"], how="left")
    u["mom_12_1"] = (1 + u["mom_12m"]) / (1 + u["mom_1m"]) - 1
    u["month"] = u["date"].dt.to_period("M")
    u = u.merge(me, on=["permno", "month"], how="left").merge(beta_m, on=["permno", "month"], how="left")
    nxt = monthly[["permno", "month_end", "ret_m", "delisted_in_month"]].assign(
        month=monthly["month_end"].dt.to_period("M") - 1).drop(columns="month_end")
    u = u.merge(nxt.rename(columns={"ret_m": "ret_next_1m", "delisted_in_month": "delisted_next_month"}),
                on=["permno", "month"], how="left")
    u["ticker"] = u["ticker"].fillna(u["ticker_snapshot"])
    return u


# ---------- Fundamentals and sectors ----------

def latest_quarters(fq: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop quarters reported after a newer quarter of the same permno, so the as-of merge never moves back."""
    fq = fq.sort_values(["permno", "report_date", "period_end"]).reset_index(drop=True)
    newest = fq.groupby("permno")["period_end"].cummax()
    keep = fq["period_end"] >= newest
    return fq[keep], int((~keep).sum())


def sic_to_sector(sic: pd.Series) -> pd.Series:
    """The fixed SIC fallback mapped to GICS sector names (first matching rule wins)."""
    out = pd.Series(np.nan, index=sic.index, dtype=object)
    for name, ranges in SIC_TO_GICS_SECTOR:
        hit = out.isna() & np.logical_or.reduce([sic.between(lo, hi) for lo, hi in ranges])
        out[hit] = name
    out[sic.isna() | (sic == 0) | sic.between(9900, 9999)] = SIC_UNCLASSIFIED
    return out.fillna("Industrials")


def add_fundamentals(u: pd.DataFrame, fq: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fq, late = latest_quarters(fq)
    cols = ["permno", "gvkey", "period_end", "report_date", "is_usd", "gsector", "ggroup", "gind", "gsubind",
            *BLOCKS, *STATEMENT_FACTORS, *PRICE_BASED_FACTORS]
    f = fq[cols].rename(columns={"period_end": "fund_period_end", "report_date": "fund_report_date",
                                 "is_usd": "fund_is_usd", **{c: f"{c}_qe" for c in PRICE_BASED_FACTORS}})
    u = pd.merge_asof(u.sort_values("date"), f.sort_values("fund_report_date"), left_on="date",
                      right_on="fund_report_date", by="permno", direction="backward")
    stale = (u["date"] - u["fund_period_end"]).dt.days > STALE_FUNDAMENTALS_DAYS
    u.loc[stale, [*BLOCKS, *STATEMENT_FACTORS, *QE_FACTORS]] = np.nan
    u["fund_stale"] = stale
    # GICS: the fundamentals row used at t, else the permno's latest row with GICS; else SIC mapped to GICS names
    gics = ["gsector", "ggroup", "gind", "gsubind"]
    latest = fq.dropna(subset=["gsector"]).sort_values("report_date").groupby("permno")[gics].last()
    for c in gics:
        u[c] = u[c].astype("Float64").fillna(u["permno"].map(latest[c]).astype("Float64")).astype("Int64")
    u["gics_from_sic"] = u["gsector"].isna()
    u["gsector_name"] = u["gsector"].map(GICS_SECTORS).astype(object)
    u["gind_name"] = u["gind"].map(GICS_INDUSTRIES).astype(object)
    u["sector_name"] = u["gsector_name"].where(~u["gics_from_sic"], sic_to_sector(u["siccd"]))
    return u, {"late_filed_quarters_dropped": late, "stale_rows": int(stale.sum())}


def add_price_scaled(u: pd.DataFrame) -> pd.DataFrame:
    ok = u["fund_is_usd"].eq(True) & (u["prc"] >= PRICE_FLOOR)
    cmc = u["company_mkt_cap_musd"].where(ok)
    u["ev"] = (cmc + u["ev_claims"]).where(lambda e: e > 0)
    u["f_book_to_price"] = u["book_equity"].where(u["book_equity"] > 0) / cmc
    u["f_earnings_to_price"] = u["ib_ttm"] / cmc
    u["f_sales_to_price"] = u["revenue_ttm"] / cmc
    u["f_dividend_yield"] = u["dv_ttm"] / cmc
    u["f_fcf_yield"] = u["fcf_ttm"] / cmc
    u["f_ebitda_to_ev"] = u["ebitda_ttm"] / u["ev"]
    u["f_ebit_to_ev"] = u["ebit_ttm"] / u["ev"]
    u["f_sales_to_ev"] = u["revenue_ttm"] / u["ev"]
    u.loc[~ok, list(QE_FACTORS)] = np.nan
    return u


# ---------- Analyst data ----------

def _ns(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """merge_asof needs the same datetime unit on both sides."""
    return df.assign(**{c: df[c].astype("datetime64[ns]") for c in cols})


def add_estimates(u: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    """For each (measure, fpi) series: the latest statpers <= t, at most 60 days old."""
    u = _ns(u, "date").sort_values("date")
    u["cfacpr_last"] = u["permno"].map(est.groupby("permno")["cfacpr_last"].first())
    for name, (measure, fpi) in SERIES.items():
        s = _ns(est[(est["measure"] == measure) & (est["fpi"] == fpi)], "statpers")
        keep = ["meanest", "stdev", "numest", "numup", "numdown"] if name == "eps1" else ["meanest"]
        s = s[["permno", "statpers", *keep]].rename(columns={"statpers": f"{name}_statpers",
                                                             **{c: f"{name}_{c}" for c in keep}})
        u = pd.merge_asof(u, s.sort_values(f"{name}_statpers"), left_on="date", right_on=f"{name}_statpers",
                          by="permno", direction="backward", tolerance=pd.Timedelta(days=IBES_MAX_AGE_DAYS))
    return u


def surprise_events(est: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """One row per EPS FY1 announcement: (actual - the last consensus before it), both on the CRSP share basis of
    the last trading day before the announcement, divided by that day's price (at most 10 days earlier). Two fiscal
    years announced the same day: the later one wins."""
    e = _ns(est[(est["measure"] == "EPS") & (est["fpi"] == 1) & est["anndats_act"].notna()], "statpers",
            "anndats_act")
    pre = e[e["statpers"] < e["anndats_act"]].sort_values("statpers")
    ev = pre.groupby(["permno", "fpedats"]).tail(1)[["permno", "fpedats", "anndats_act", "actual", "meanest",
                                                     "cfacpr_last"]]
    ev = ev.sort_values(["permno", "anndats_act", "fpedats"]).drop_duplicates(["permno", "anndats_act"], keep="last")
    px = _ns(prices.dropna(subset=["prc", "cfacpr"]), "date")
    px = px[px["cfacpr"] > 0].rename(columns={"date": "price_date_ann", "prc": "prc_ann", "cfacpr": "cfacpr_ann"})
    ev = pd.merge_asof(ev.sort_values("anndats_act"), px.sort_values("price_date_ann"), left_on="anndats_act",
                       right_on="price_date_ann", by="permno", direction="backward", allow_exact_matches=False,
                       tolerance=pd.Timedelta(days=SURPRISE_PRICE_MAX_DAYS))
    basis = ev["cfacpr_ann"] / ev["cfacpr_last"]
    ev["f_analyst_surprise"] = ((ev["actual"] - ev["meanest"]) * basis
                                / ev["prc_ann"].where(ev["prc_ann"] >= PRICE_FLOOR))
    return ev.rename(columns={"anndats_act": "surprise_anndats", "fpedats": "surprise_fpedats"})[
        ["permno", "surprise_anndats", "surprise_fpedats", "f_analyst_surprise"]]


def add_analyst(u: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    u = pd.merge_asof(_ns(u, "date").sort_values("date"), _ns(events, "surprise_anndats").sort_values(
                      "surprise_anndats"), left_on="date",
                      right_on="surprise_anndats", by="permno", direction="backward",
                      tolerance=pd.Timedelta(days=SURPRISE_MAX_AGE_DAYS))
    basis = u["cfacpr_t"] / u["cfacpr_last"]                      # IBES per-share values -> shares at t
    prc = u["prc"].where(u["prc"] >= PRICE_FLOOR)
    u["eps1_crsp"], u["eps2_crsp"] = u["eps1_meanest"] * basis, u["eps2_meanest"] * basis
    u["f_forward_ep"] = u["eps1_crsp"] / prc
    u["f_forward_ep_fy2"] = u["eps2_crsp"] / prc
    u["f_forward_cfp"] = u["cps1_meanest"] * basis / prc
    u["f_forward_sales_to_ev"] = u["sal1_meanest"] / u["ev"]
    u["f_forward_ebitda_to_ev"] = u["ebt1_meanest"] / u["ev"]
    u["f_ltg"] = u["ltg_meanest"]
    u["f_eps_revision"] = (u["eps1_numup"] - u["eps1_numdown"]) / u["eps1_numest"]
    u["f_eps_dispersion"] = (u["eps1_stdev"] / u["eps1_meanest"].abs()).where(
        (u["eps1_meanest"].abs() >= DISPERSION_MIN_MEAN) & (u["eps1_numest"] >= DISPERSION_MIN_NUMEST))
    u = u.rename(columns={"eps1_numest": "numest_fy1", "eps1_statpers": "ibes_statpers"})
    u[list(ANALYST)] = u[list(ANALYST)].replace([np.inf, -np.inf], np.nan)
    return u


# ---------- RV multiples and peers ----------

def _pos(s: pd.Series) -> pd.Series:
    return s.where(s > 0)


def build_rv(u: pd.DataFrame) -> pd.DataFrame:
    cmc = u["company_mkt_cap_musd"].where(u["fund_is_usd"].eq(True) & (u["prc"] >= PRICE_FLOOR))
    rv = u[["date", "permno", "gsector", "gind", "prc", "company_mkt_cap_musd", "ev"]].copy()
    rv["pe"], rv["pb"], rv["ps"] = cmc / _pos(u["ib_ttm"]), cmc / _pos(u["book_equity"]), cmc / _pos(u["revenue_ttm"])
    rv["ev_ebitda"] = u["ev"] / _pos(u["ebitda_ttm"])
    rv["ev_ebit"] = u["ev"] / _pos(u["ebit_ttm"])
    rv["ev_sales"] = u["ev"] / _pos(u["revenue_ttm"])
    rv["fcf_yield"], rv["dividend_yield"] = u["fcf_ttm"] / cmc, u["dv_ttm"] / cmc
    rv["earnings_yield"] = u["ib_ttm"] / cmc
    prc = u["prc"].where(u["prc"] >= PRICE_FLOOR)
    rv["fwd_pe_fy1"], rv["fwd_pe_fy2"] = prc / _pos(u["eps1_crsp"]), prc / _pos(u["eps2_crsp"])
    rv["fwd_ev_ebitda"] = u["ev"] / _pos(u["ebt1_meanest"])
    rv["fwd_ev_sales"] = u["ev"] / _pos(u["sal1_meanest"])
    rv["ltg"] = u["ltg_meanest"]
    rv["peg"] = rv["fwd_pe_fy1"] / (_pos(rv["ltg"]) * 100)
    rv[list(MULTIPLES)] = rv[list(MULTIPLES)].replace([np.inf, -np.inf], np.nan)
    for group in ("gind", "gsector"):
        g = rv.groupby(["date", group])
        for c in MULTIPLES:
            rv[f"{c}_pct_{group}"] = g[c].rank(pct=True).where(g[c].transform("count") >= PEER_MIN)
    return rv


def build_peers(rv: pd.DataFrame) -> pd.DataFrame:
    g = rv.dropna(subset=["gind"]).groupby(["date", "gind"])
    out = g.size().rename("n_peers").to_frame()
    for c in MULTIPLES:
        q = g[c].quantile([0.25, 0.5, 0.75]).unstack()
        n = g[c].count()
        out[f"{c}_n"] = n
        for p, col in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
            out[f"{c}_{col}"] = q[p].where(n >= PEER_MIN)
    out = out.reset_index()
    out.insert(2, "gind_name", out["gind"].map(GICS_INDUSTRIES).astype(object))
    return out


def build_master(listed, monthly, fq, link, panel, snaps) -> pd.DataFrame:
    mo = monthly.sort_values("month_end")
    last = mo.groupby("permno").agg(permco=("permco", "last"), ticker=("ticker", "last"), siccd=("siccd", "last"),
                                    first_price_date=("last_trade_date", "min"),
                                    last_price_date=("last_trade_date", "max"))
    gics = fq.dropna(subset=["gsector"]).sort_values("report_date").groupby("permno")[
        ["gsector", "ggroup", "gind", "gsubind"]].last()
    gvkey = fq.sort_values("report_date").groupby("permno")["gvkey"].last()
    names = snaps.sort_values("report_dt").groupby("permno")["security_name"].last()
    tickers = link.sort_values("sdate").groupby("permno")["ticker"].agg(lambda s: ";".join(dict.fromkeys(s)))
    uni = panel.groupby("permno")["date"].agg(first_universe_date="min", last_universe_date="max")
    excl = snaps.groupby("permno")["in_universe"].agg(lambda s: bool((~s).any()))
    m = pd.DataFrame(index=pd.Index(sorted(listed), name="permno")).join(last).join(gics).join(uni)
    m["gvkey"] = m.index.map(gvkey)
    m["ibes_ticker"] = m.index.map(tickers)
    m["security_name"] = m.index.map(names)
    m["excluded_non_member"] = m.index.map(excl).fillna(False).astype(bool)
    for c in ("gsector", "ggroup", "gind", "gsubind"):
        m[c] = m[c].astype("Int64")
    m["gsector_name"] = m["gsector"].map(GICS_SECTORS).astype(object)
    m["gind_name"] = m["gind"].map(GICS_INDUSTRIES).astype(object)
    cols = ["permco", "gvkey", "ibes_ticker", "ticker", "security_name", "gsector", "gsector_name", "ggroup", "gind",
            "gind_name", "gsubind", "siccd", "first_price_date", "last_price_date", "first_universe_date",
            "last_universe_date", "excluded_non_member"]
    return m[cols].reset_index()


# ---------- Checks ----------

def coverage_by_year(df: pd.DataFrame, cols) -> dict:
    y = df["date"].dt.year
    return {c: {str(k): round(float(v), 3) for k, v in df[c].notna().groupby(y).mean().items()} for c in cols}


def share_class_gap(u: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """By year: company shares that Compustat counts but our permnos do not. On the quarter-end date of cshoq,
    Compustat cshoq (millions) vs the summed CRSP shrout (thousands / 1000) of our permnos with the same permco in that
    month. A company counts when the gap is above 5% of its Compustat shares; the missing market cap at t is the
    missing share fraction applied to the company market cap at t."""
    crsp = monthly.groupby(["permco", monthly["month_end"].dt.to_period("M")])["shrout"].sum() / 1000
    c = u.dropna(subset=["permco", "cshoq", "fund_period_end", "company_mkt_cap_musd"]).drop_duplicates(
        ["date", "permco"])
    c = c[c["cshoq"] > 0]
    ours = crsp.reindex(pd.MultiIndex.from_arrays([c["permco"], c["fund_period_end"].dt.to_period("M")])).to_numpy()
    share_gap = np.clip((c["cshoq"].to_numpy() - ours) / c["cshoq"].to_numpy(), 0, 1)
    big = share_gap > SHARE_CLASS_GAP_COMPANY
    missing = np.where(big, c["company_mkt_cap_musd"] * share_gap / np.maximum(1 - share_gap, 1e-9), 0.0)
    year = c["date"].dt.year
    firms = u.drop_duplicates(["date", "permco"])
    total = firms.groupby(firms["date"].dt.year)["company_mkt_cap_musd"].sum()
    out = pd.DataFrame({"missing_mkt_cap_musd": pd.Series(missing, index=c.index).groupby(year).sum(),
                        "company_months": pd.Series(big, index=c.index).groupby(year).sum(),
                        "universe_mkt_cap_musd": total})
    out["share"] = out["missing_mkt_cap_musd"] / out["universe_mkt_cap_musd"]
    return out


def run_checks(panel: pd.DataFrame, rv: pd.DataFrame, snaps: pd.DataFrame,
               monthly: pd.DataFrame) -> tuple[list[Check], dict]:
    n = panel.groupby("date").size()
    lo, hi = SNAPSHOT_SIZE_RANGE
    checks = [check("no duplicate (date, permno)", not panel.duplicated(["date", "permno"]).any()),
              check(f"stocks per month within {lo:,}-{hi:,}", bool(n.between(lo, hi).all()),
                    f"{n.min():,} ({n.idxmin():%Y-%m}) to {n.max():,} ({n.idxmax():%Y-%m})", level=WARN)]
    for col, name in (("fund_report_date", "fundamentals reported after t"),
                      ("ibes_statpers", "IBES statpers after t"), ("snapshot_eff_dt", "snapshot effective after t"),
                      ("surprise_anndats", "surprise announced after t"), ("price_date", "price after t")):
        checks.append(check(f"look-ahead: no {name}", not (panel[col] > panel["date"]).any()))
    old = pd.Timedelta(days=IBES_MAX_AGE_DAYS)
    for name in SERIES:
        sp = panel["ibes_statpers" if name == "eps1" else f"{name}_statpers"]
        checks.append(check(f"look-ahead: {name} statpers on or before t and at most {IBES_MAX_AGE_DAYS} days old",
                            not ((sp > panel["date"]) | (sp < panel["date"] - old)).any()))
    excluded = snaps.loc[~snaps["in_universe"], ["report_dt", "permno"]].rename(
        columns={"report_dt": "snapshot_report_dt"})
    leak = panel.merge(excluded, on=["snapshot_report_dt", "permno"])
    checks.append(check("excluded non-member stock-snapshots are not in the panel", leak.empty, f"{len(leak)} rows"))
    cm = panel.dropna(subset=["company_mkt_cap_musd", "mkt_cap_musd"])
    checks.append(check("company market cap >= the stock's market cap",
                        bool((cm["company_mkt_cap_musd"] >= cm["mkt_cap_musd"] - 1e-9).all())))
    no_price = ~panel["fund_is_usd"].eq(True) | ~(panel["prc"] >= PRICE_FLOOR)
    leaked = int(panel.loc[no_price, list(PRICE_SCALED)].notna().sum().sum())
    checks.append(check("no price-scaled factor on non-USD fundamentals or prices below $0.01", leaked == 0))
    stale_vals = int(panel.loc[panel["fund_stale"], list(STATEMENT_FACTORS)].notna().sum().sum())
    checks.append(check("stale fundamentals (> 365 days) have no factor values", stale_vals == 0))
    checks.append(check("every row has a sector name", bool(panel["sector_name"].notna().all())))

    gaps = share_class_gap(panel, monthly)
    bad = gaps[gaps["share"] > SHARE_CLASS_GAP_YEAR]
    checks.append(check(f"share-class gap at most {SHARE_CLASS_GAP_YEAR:.0%} of universe market cap in every year",
                        bad.empty, f"max {gaps['share'].max():.2%} ({gaps['share'].idxmax()})"
                        + (f"; above: {bad['share'].round(4).to_dict()}" if len(bad) else ""), level=WARN))
    med = rv.groupby(rv["date"].dt.year)[list(SANITY_MEDIAN_RANGES)].median()
    outside = {c: [int(y) for y, v in med[c].items() if not lo_ <= v <= hi_]
               for c, (lo_, hi_) in SANITY_MEDIAN_RANGES.items()}
    checks.append(check("median pe, pb, ev_ebitda by year in the expected small-cap ranges",
                        not any(outside.values()),
                        "; ".join(f"{c} outside {SANITY_MEDIAN_RANGES[c]} in {v}" for c, v in outside.items() if v)
                        or "all years inside", level=WARN))
    extra = {"share_class_gap_by_year": {str(k): {"share": round(float(r.share), 5),
                                                  "company_months": int(r.company_months)}
                                         for k, r in gaps.iterrows()},
             "median_multiples_by_year": {c: {str(y): round(float(v), 2) for y, v in med[c].items()}
                                          for c in SANITY_MEDIAN_RANGES}}
    return checks, extra


# ---------- Stage ----------

PANEL_COLS = ("date", "permno", "permco", "gvkey", "ticker", "security_name", "gsector", "gsector_name", "ggroup",
              "gind", "gind_name", "gsubind", "sector_name", "gics_from_sic", "siccd", "index_weight",
              "snapshot_report_dt", "snapshot_eff_dt", "new_entrant", "price_date", "prc", "shrout", "mkt_cap_musd",
              "company_mkt_cap_musd", "n_share_classes", "cfacpr_t", "cfacpr_last", *PRICE_FACTORS, "beta_252d",
              "beta_60m", "fund_period_end", "fund_report_date", "fund_is_usd", "fund_stale", *BLOCKS, "ev",
              *STATEMENT_FACTORS, *PRICE_SCALED, *QE_FACTORS, "ibes_statpers", "numest_fy1", *ANALYST,
              "surprise_anndats", "surprise_fpedats", "ret_next_1m", "delisted_next_month",
              *(f"{s}_statpers" for s in SERIES if s != "eps1"))


def build(paths: DataPaths) -> dict:
    s01, s02, s03 = (paths.stage_dir(s) for s in ("s01_membership", "s02_prices", "s03_fundamentals_q"))
    s05, s06 = paths.stage_dir("s05_ibes"), paths.stage_dir("s06_market")
    snaps = pd.read_parquet(s01 / "snapshots.parquet")
    monthly = pd.read_parquet(s02 / "monthly.parquet")
    calendar = pd.read_parquet(s02 / "calendar.parquet")
    mkt_d = pd.read_parquet(s06 / "market_daily.parquet", columns=["date", "mktrf", "rf"])
    mkt_m = pd.read_parquet(s06 / "market_monthly.parquet", columns=["month", "mktrf", "rf"])
    est = pd.read_parquet(s05 / "estimates_consensus.parquet",
                          columns=["permno", "statpers", "measure", "fpi", "fpedats", "meanest", "stdev", "numest",
                                   "numup", "numdown", "actual", "anndats_act", "cfacpr_last"],
                          filters=[("measure", "in", ["EPS", "CPS", "SAL", "EBT"]), ("fpi", "in", [0, 1, 2])])
    dates = rebalance_dates(calendar, snaps)
    u = build_universe(dates, snaps)
    daily = pd.read_parquet(s02 / "daily", columns=["date", "permno", "prc", "ret", "cfacpr", "is_delisting_row"])
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    me = daily_month_end(daily, mkt_d)
    events = surprise_events(est, daily[["date", "permno", "prc", "cfacpr"]])
    del daily
    u = add_market(u, monthly, me, monthly_beta(monthly, mkt_m))
    fq = pd.read_parquet(s03 / "fundamentals_quarterly.parquet")
    u, finfo = add_fundamentals(u, fq)
    u = add_price_scaled(u)
    u = add_analyst(add_estimates(u, est), events)
    del est
    u = u.sort_values(["date", "permno"]).reset_index(drop=True)
    rv = build_rv(u)
    listed = set(int(x) for x in (s01 / "permno_list.txt").read_text().split())
    master = build_master(listed, monthly, fq, pd.read_parquet(s05 / "link.parquet"), u, snaps)
    return {"panel": u[list(PANEL_COLS)], "rv": rv, "peers": build_peers(rv), "master": master, "snaps": snaps,
            "monthly": monthly, "info": finfo}


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    b = build(paths)
    panel, rv, peers, master = b["panel"], b["rv"], b["peers"], b["master"]
    checks, cinfo = run_checks(panel, rv, b["snaps"], b["monthly"])
    write_parquet(panel, out / PANEL)
    write_parquet(rv, out / RV)
    write_parquet(peers, out / PEERS)
    write_parquet(master, out / MASTER)

    factor_cols = ["prc", "mkt_cap_musd", "company_mkt_cap_musd", *PRICE_FACTORS, "beta_252d", "beta_60m",
                   *STATEMENT_FACTORS, *PRICE_SCALED, *QE_FACTORS, *ANALYST, "ret_next_1m"]
    n = panel.groupby("date").size()
    extra = {**b["info"], **cinfo, "months": int(panel["date"].nunique()),
             "stocks_per_month": {"min": int(n.min()), "median": float(n.median()), "max": int(n.max())},
             "sector_from_sic_rows": int(panel["gics_from_sic"].sum()),
             "coverage_by_year": coverage_by_year(panel, ["prc", "f_roa", "f_book_to_price", "f_forward_ep",
                                                          "beta_252d", "ret_next_1m"]),
             "rv_coverage_by_year": coverage_by_year(rv, list(MULTIPLES)),
             "f_forward_ep_p0_1_p99_9": [round(float(panel["f_forward_ep"].quantile(q)), 4) for q in (0.001, 0.999)]}
    tables = {PANEL: table_info(panel, "date", factor_cols), RV: table_info(rv, "date", list(MULTIPLES)),
              PEERS: table_info(peers, "date", ["n_peers"]), MASTER: table_info(master, None, [])}
    summary = write_summary(out, STAGE, tables, checks, extra)
    print(f"  panel {len(panel):,} rows, {extra['months']} months {panel['date'].min():%Y-%m} to "
          f"{panel['date'].max():%Y-%m}, {n.min():,}-{n.max():,} stocks | stale fundamentals {b['info']['stale_rows']:,}"
          f" | late-filed quarters dropped {b['info']['late_filed_quarters_dropped']} | peer rows {len(peers):,} | "
          f"master {len(master):,}")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
