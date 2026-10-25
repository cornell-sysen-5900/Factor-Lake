"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s06_market.py
PURPOSE: Stage s06_market: daily and monthly market series: Russell 2000 / Growth / Value price returns, IWM / IWO /
         IWN total returns, Fama-French factors and rf, FRED yields and spreads.

In:  raw/index_close.csv, raw/benchmark_etfs.csv, raw/ff_daily.csv, raw/ff_monthly.csv, raw/fred_*.csv,
     processed/s02_prices/calendar.parquet (month ends)
Out: processed/s06_market/market_daily.parquet    one row per date (trading day or FRED business day)
     processed/s06_market/market_monthly.parquet  one row per month (month_end = last CRSP trading day)
     processed/s06_market/summary.json

Rules (pipeline/PLAN.md, s06_market):
    - index levels: 2012-10-29 and 2012-10-30 dropped (Hurricane Sandy, no CRSP rows); price return from consecutive
      levels; monthly = level at month end vs the previous month end
    - ETF daily total returns from CRSP; monthly = compounded daily returns
    - Fama-French: already decimals; monthly = the published monthly factors, matched by the month of dateff
    - FRED: percent / 100; blank days (holidays) forward-filled and flagged in was_filled;
      term_spread = 10-year - 3-month, credit_spread = Baa - Aaa; monthly = value on the last day <= month end
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check
from pipeline.config import (CLOSED_MARKET_DAYS, DAILY_START, ETF_PERMNOS, FF_MONTHLY_TOLERANCE, FRED_SERIES,
                             INDEX_TICS, IWM_MINUS_INDEX_RANGE, MAX_FILLED_RUN_DAYS, YIELD_RANGES, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary

STAGE = "s06_market"
FF_COLS = ("mktrf", "smb", "hml", "rmw", "cma", "umd", "rf")
INDEX_COLS = tuple(f"{n}_price_ret" for n in INDEX_TICS.values())
ETF_COLS = tuple(f"{n}_total_ret" for n in ETF_PERMNOS.values())
YIELD_COLS = tuple(FRED_SERIES.values())
SPREADS = ("term_spread", "credit_spread")
RETURN_COLS = INDEX_COLS + ETF_COLS + FF_COLS
MONTHLY_COLS = ("month", "month_end") + RETURN_COLS + YIELD_COLS + SPREADS


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    return df


def index_levels(path) -> pd.DataFrame:
    """date x {r2000, r2000_growth, r2000_value} levels, market-closed days dropped."""
    idx = _lower(pd.read_csv(path, parse_dates=["datadate"]))
    idx = idx[~idx["datadate"].isin(pd.to_datetime(list(CLOSED_MARKET_DAYS)))]
    lvl = idx.pivot(index="datadate", columns="tic", values="prccd").rename(columns=INDEX_TICS).sort_index()
    lvl.index.name = "date"
    return lvl[list(INDEX_TICS.values())]


def etf_returns(path) -> pd.DataFrame:
    etf = _lower(pd.read_csv(path))
    etf["dlycaldt"] = pd.to_datetime(etf["dlycaldt"])
    r = etf.pivot(index="dlycaldt", columns="permno", values="dlyret").rename(
        columns={p: f"{n}_total_ret" for p, n in ETF_PERMNOS.items()})
    r.index.name = "date"
    return r[list(ETF_COLS)].sort_index()


def fama_french(daily_path, monthly_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = _lower(pd.read_csv(daily_path, parse_dates=["date"])).set_index("date")[list(FF_COLS)].sort_index()
    m = _lower(pd.read_csv(monthly_path, parse_dates=["dateff"]))
    m["month"] = m["dateff"].dt.to_period("M")
    return d, m.set_index("month")[list(FF_COLS)]


def fred_yields(paths: DataPaths) -> pd.DataFrame:
    """FRED series in decimals, holidays forward-filled; was_filled = any series filled on that day."""
    cols = {}
    for series, name in FRED_SERIES.items():
        f = _lower(pd.read_csv(paths.raw / f"fred_{series}.csv", parse_dates=["observation_date"],
                               na_values=["", "."]))
        cols[name] = f.set_index("observation_date")[series.lower()] / 100
    y = pd.DataFrame(cols).sort_index()
    y.index.name = "date"
    blank = y.isna()
    y = y.ffill()
    y["was_filled"] = (blank & y.notna()).any(axis=1)
    return y


def build_daily(lvl, etf, ff_d, yields) -> pd.DataFrame:
    daily = (lvl.pct_change(fill_method=None).rename(columns=lambda c: f"{c}_price_ret")
             .join(etf, how="outer").join(ff_d, how="outer"))
    daily = daily.join(yields, how="outer")
    daily[list(YIELD_COLS)] = daily[list(YIELD_COLS)].ffill()          # trading days FRED does not list
    daily["was_filled"] = daily["was_filled"].astype("boolean").fillna(True).astype(bool) & daily[
        list(YIELD_COLS)].notna().any(axis=1)
    daily["term_spread"] = daily["ust_10y"] - daily["tbill_3m"]
    daily["credit_spread"] = daily["moodys_baa"] - daily["moodys_aaa"]
    daily = daily[daily.index >= pd.Timestamp(DAILY_START)]
    daily = daily.iloc[1:] if daily[list(INDEX_COLS)].iloc[0].isna().all() else daily   # first day has no return
    return daily.reset_index()[["date", *RETURN_COLS, *YIELD_COLS, *SPREADS, "was_filled"]]


def build_monthly(lvl, etf, ff_m, daily, calendar) -> pd.DataFrame:
    """Months of the CRSP calendar; the first month (no previous level) is dropped."""
    cal = calendar.assign(month=pd.PeriodIndex(calendar["month"], freq="M")).set_index("month")["month_end"]
    month_lvl = pd.DataFrame({m: lvl[lvl.index <= end].iloc[-1] for m, end in cal.items()
                              if (lvl.index <= end).any()}).T
    out = pd.DataFrame(index=cal.index)
    out["month_end"] = cal
    out = out.join(month_lvl.pct_change(fill_method=None).rename(columns=lambda c: f"{c}_price_ret"))
    etf_m = (1 + etf).groupby(etf.index.to_period("M")).prod(min_count=1) - 1
    out = out.join(etf_m).join(ff_m)
    d = daily.set_index("date")
    for c in (*YIELD_COLS, *SPREADS):
        out[c] = [d.loc[d.index <= end, c].dropna().iloc[-1] if (d.index <= end).any() else np.nan
                  for end in out["month_end"]]
    out = out.iloc[1:].reset_index(names="month")
    out["month"] = out["month"].astype(str)
    return out[list(MONTHLY_COLS)]


def run_checks(daily, monthly, ff_d) -> list[Check]:
    checks = [check("market-closed days dropped from the index returns",
                    not daily["date"].isin(pd.to_datetime(list(CLOSED_MARKET_DAYS))).any()
                    or daily.loc[daily["date"].isin(pd.to_datetime(list(CLOSED_MARKET_DAYS))),
                                 list(INDEX_COLS)].isna().all().all())]
    for c, (lo, hi) in YIELD_RANGES.items():
        v = daily[c].dropna()
        checks.append(check(f"{c} within {lo:.1%}-{hi:.0%}", bool(v.between(lo, hi).all()),
                            f"{v.min():.2%} to {v.max():.2%}"))
    checks.append(check("Fama-French daily rf in decimals (mean below 0.001)", bool(ff_d["rf"].mean() < 0.001),
                        f"mean {ff_d['rf'].mean():.6f}"))
    both = monthly.dropna(subset=["iwm_total_ret", "r2000_price_ret"])
    gap = float((both["iwm_total_ret"] - both["r2000_price_ret"]).mean() * 12)
    lo, hi = IWM_MINUS_INDEX_RANGE
    checks.append(check(f"IWM total minus Russell 2000 price return {lo:.1%}-{hi:.1%} a year", lo <= gap <= hi,
                        f"{gap:.2%} a year over {len(both)} months", level=WARN))
    # published monthly Fama-French vs compounded daily (mktrf: compounded market minus compounded rf)
    g = ff_d.groupby(ff_d.index.to_period("M"))
    comp = (1 + ff_d).groupby(ff_d.index.to_period("M")).prod() - 1
    comp["mktrf"] = ((1 + ff_d["mktrf"] + ff_d["rf"]).groupby(ff_d.index.to_period("M")).prod() - 1) - comp["rf"]
    comp = comp[g.size() >= 15]                                           # complete months only
    pub = monthly.assign(month=pd.PeriodIndex(monthly["month"], freq="M")).set_index("month")[list(FF_COLS)]
    diff = (pub - comp).abs().dropna(how="all")
    worst = {c: f"max {diff[c].max() * 1e4:.0f} bp, {int((diff[c] > FF_MONTHLY_TOLERANCE).sum())} months over"
             for c in FF_COLS}
    # strict for the market and rf; the long-short factors' daily portfolios are built differently (informational)
    strict, info = ("mktrf", "rf"), ("smb", "hml", "rmw", "cma", "umd")
    over = int((diff[list(strict)] > FF_MONTHLY_TOLERANCE).sum().sum())
    checks.append(check(f"published monthly mktrf and rf within {FF_MONTHLY_TOLERANCE * 1e4:.0f} bp of compounded "
                        "daily", over == 0, "; ".join(f"{k}: {worst[k]}" for k in strict)))
    checks.append(check("published monthly smb, hml, rmw, cma, umd vs compounded daily (informational)", True,
                        "; ".join(f"{k}: {worst[k]}" for k in info)))
    run = daily["was_filled"].astype(int).groupby((~daily["was_filled"]).cumsum()).sum().max()
    checks.append(check(f"no yield filled for more than {MAX_FILLED_RUN_DAYS} days in a row",
                        int(run) <= MAX_FILLED_RUN_DAYS, f"longest run {int(run)} days", level=WARN))
    months = pd.PeriodIndex(monthly["month"], freq="M")
    checks.append(check("monthly table has every month", len(months) == (months.max() - months.min()).n + 1,
                        f"{months.min()} to {months.max()}"))
    return checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    lvl = index_levels(paths.raw / "index_close.csv")
    etf = etf_returns(paths.raw / "benchmark_etfs.csv")
    ff_d, ff_m = fama_french(paths.raw / "ff_daily.csv", paths.raw / "ff_monthly.csv")
    yields = fred_yields(paths)
    calendar = pd.read_parquet(paths.stage_dir("s02_prices") / "calendar.parquet")
    daily = build_daily(lvl, etf, ff_d, yields)
    monthly = build_monthly(lvl, etf, ff_m, daily, calendar)
    checks = run_checks(daily, monthly, ff_d)
    write_parquet(daily, out / "market_daily.parquet")
    write_parquet(monthly, out / "market_monthly.parquet")

    ends = {c: str(daily.loc[daily[c].notna(), "date"].max().date()) for c in (*RETURN_COLS, *YIELD_COLS)}
    stat_cols = [*RETURN_COLS, *YIELD_COLS, *SPREADS]
    extra = {"source_ends": ends, "filled_days": int(daily["was_filled"].sum()),
             "coverage_daily": {c: round(float(daily[c].notna().mean()), 4) for c in stat_cols},
             "coverage_monthly": {c: round(float(monthly[c].notna().mean()), 4) for c in stat_cols}}
    tables = {"market_daily.parquet": table_info(daily, "date", stat_cols),
              "market_monthly.parquet": table_info(monthly, "month_end", stat_cols)}
    summary = write_summary(out, STAGE, tables, checks, extra)
    print(f"  daily {len(daily):,} rows {daily['date'].min().date()} to {daily['date'].max().date()} "
          f"({extra['filled_days']} with filled yields) | monthly {len(monthly)} months {monthly['month'].iloc[0]} to "
          f"{monthly['month'].iloc[-1]}")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
