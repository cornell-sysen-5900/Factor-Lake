"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s02_prices.py
PURPOSE: Stage s02_prices: clean the CRSP CIZ daily file, add momentum and volatility, build monthly returns.

In:  raw/r2000_price_crsp_ciz.csv (15 columns), raw/delistings.csv, raw/query_inputs/permno_list.txt (to compare)
Out: processed/s02_prices/daily/year=YYYY/data.parquet   one row per permno per trading day
     processed/s02_prices/monthly.parquet                 one row per permno per month (returns + month-end data)
     processed/s02_prices/calendar.parquet                month, month_end (last CRSP trading day), trading_days
     processed/s02_prices/delistings.parquet              one row per delisting, with the return used
     processed/s02_prices/summary.json

Rules (pipeline/PLAN.md, s02_prices):
    - exact duplicate rows dropped; conflicting duplicates FAIL
    - prc = |DlyPrc|, 0 = missing; ret < -1 = missing; vol < 0, shrout <= 0 = missing; mkt_cap = prc x shrout ($k)
    - delisting rows (DlyDelFlg = Y) already hold the delisting return; a missing one is filled -30% for GDR,
      else 0%. They never provide a price or market cap (prc, prc_adj, mkt_cap are missing on them).
    - momentum and volatility from compounded returns over trading-row windows, 80% of rows required
    - monthly return compounds every row of the month; month-end market data come from the last real trading day
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from pipeline.checks import PASS, WARN, Check, check, vs_last_run
from pipeline.config import (DELISTING_FILL, DELISTING_FILL_DEFAULT, MIN_OBS_SHARE, MOMENTUM_WINDOWS,
                             PARQUET_COMPRESSION, TRADING_DAYS_PER_YEAR, VOL_WINDOWS, DataPaths)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages.s00_inventory import read_header

STAGE = "s02_prices"
CIZ_FILE = "r2000_price_crsp_ciz.csv"
_NUMERIC = ("dlyprc", "dlyret", "dlyvol", "shrout", "dlyopen", "dlyhigh", "dlylow", "dlycumfacpr")
ALL_COLUMNS = ("permno", "permco", "dlycaldt", "ticker", "siccd", "dlydelflg") + _NUMERIC
MARKET_CAP_COLUMNS = ("permno", "dlycaldt", "dlydelflg", "dlyprc", "shrout")       # what s01 needs
_TYPES = {"permno": pa.int32(), "permco": pa.int32(), "dlycaldt": pa.date32(), "siccd": pa.string(),
          "ticker": pa.dictionary(pa.int32(), pa.string()), "dlydelflg": pa.dictionary(pa.int32(), pa.string()),
          **{c: pa.float64() for c in _NUMERIC}}
MOM_COLS = tuple(MOMENTUM_WINDOWS)
VOL_COLS = tuple(f"vol_{kind}_{label}" for label in VOL_WINDOWS for kind in ("gk", "close"))
DAILY_COLUMNS = ("date", "permno", "permco", "ticker", "siccd", "prc", "prc_adj", "cfacpr", "shrout", "mkt_cap",
                 "mkt_cap_musd", "ret", "vol", "is_delisting_row", "delist_ret_filled") + MOM_COLS + VOL_COLS


# ---------- Reading and cleaning (also used by s01 for month-end market caps) ----------

def read_ciz(path, columns=ALL_COLUMNS) -> pd.DataFrame:
    """Read only `columns` of the CIZ daily file (names matched case-insensitively), sorted by permno and date in
    a stable way, so 'first' among duplicates means first in the file."""
    original = {c.strip().lower(): c for c in read_header(path)}
    tbl = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(
        include_columns=[original[c] for c in columns], column_types={original[c]: _TYPES[c] for c in columns}))
    df = tbl.rename_columns([c.strip().lower() for c in tbl.column_names]).to_pandas(date_as_object=False)
    df = df.rename(columns={"dlycaldt": "date"})
    df["date"] = df["date"].astype("datetime64[ns]")
    if "siccd" in df.columns:
        df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce")
    df = df.dropna(subset=["permno", "date"])
    return df.sort_values(["permno", "date"], kind="stable").reset_index(drop=True)


def drop_duplicates(px: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop repeated (permno, date) rows, keeping the first. Returns the frame, rows removed, and the number of
    keys whose repeated rows differ (those would be real conflicts, not copies)."""
    key = px.duplicated(["permno", "date"], keep=False)
    conflicts = int(px[key].drop_duplicates().duplicated(["permno", "date"]).sum())
    out = px[~px.duplicated(["permno", "date"], keep="first")].reset_index(drop=True)
    return out, len(px) - len(out), conflicts


def clean_prices(px: pd.DataFrame) -> pd.DataFrame:
    """is_delisting_row, prc, shrout and market cap. Delisting rows never carry a price or market cap."""
    px["is_delisting_row"] = (px["dlydelflg"] == "Y").to_numpy()
    px["prc"] = px["dlyprc"].abs().where(lambda s: s > 0).where(~px["is_delisting_row"])
    px["shrout"] = px["shrout"].where(px["shrout"] > 0)
    px["mkt_cap"] = px["prc"] * px["shrout"]                    # $ thousands (shrout in thousands)
    px["mkt_cap_musd"] = px["mkt_cap"] / 1000
    return px


def month_end_rows(px: pd.DataFrame) -> pd.Series:
    """True on the last real trading row (not a delisting row) of each permno and month. px is sorted."""
    real = px[~px["is_delisting_row"]]
    last = real.groupby([real["permno"], real["date"].dt.to_period("M")], sort=False).tail(1).index
    return pd.Series(px.index.isin(last), index=px.index)


def month_end_market_caps(px: pd.DataFrame) -> pd.DataFrame:
    """permno, month, last_trade_date, mkt_cap_musd at the last real trading day of each month (s01 rule 6 and
    s02's monthly table use this same function)."""
    rows = px[month_end_rows(px)]
    return pd.DataFrame({"permno": rows["permno"].to_numpy(), "month": rows["date"].dt.to_period("M").to_numpy(),
                         "last_trade_date": rows["date"].to_numpy(), "mkt_cap_musd": rows["mkt_cap_musd"].to_numpy()})


def fill_delisting_returns(px: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """ret = DlyRet (below -100% = missing code). A delisting row without a return gets -30% for GDR, else 0%."""
    px["ret"] = px["dlyret"].where(px["dlyret"] >= -1)
    action = px["permno"].map(delistings.drop_duplicates("permno").set_index("permno")["delactiontype"])
    fill = px["is_delisting_row"] & px["ret"].isna()
    px.loc[fill, "ret"] = action[fill].map(DELISTING_FILL).fillna(DELISTING_FILL_DEFAULT).astype("float64")
    px["delist_ret_filled"] = fill
    return px


# ---------- Momentum and volatility (ported from the notebook's PRICES cell) ----------

def add_momentum(px: pd.DataFrame) -> pd.DataFrame:
    """Compounded return over the last n rows; NaN unless MIN_OBS_SHARE of them have a return."""
    g = px["permno"]
    lr = np.log1p(px["ret"])
    cs = lr.fillna(0).groupby(g).cumsum()
    cn = lr.notna().astype("int32").groupby(g).cumsum()
    for name, n in MOMENTUM_WINDOWS.items():
        s = cs - cs.groupby(g).shift(n)
        k = cn - cn.groupby(g).shift(n)
        px[name] = np.where(k >= MIN_OBS_SHARE * n, np.expm1(s), np.nan)
    return px


def _rolling(x: pd.Series, g: pd.Series, n: int, want_std: bool) -> pd.Series:
    """Rolling mean or sample std over the last n rows per permno (fewer before the n-th row), via cumulative sums.
    NaN unless MIN_OBS_SHARE x n values are present."""
    s1 = x.fillna(0).groupby(g).cumsum()
    s2 = (x.fillna(0) ** 2).groupby(g).cumsum()
    c = x.notna().astype("int32").groupby(g).cumsum()
    first = c.groupby(g).shift(n).isna()                     # before the n-th row: use everything so far
    d1 = (s1 - s1.groupby(g).shift(n).fillna(0)).where(~first, s1)
    d2 = (s2 - s2.groupby(g).shift(n).fillna(0)).where(~first, s2)
    k = (c - c.groupby(g).shift(n).fillna(0)).where(~first, c)
    mean = d1 / k
    out = np.sqrt(((d2 - k * mean ** 2) / (k - 1)).clip(lower=0)) if want_std else mean
    return out.where(k >= MIN_OBS_SHARE * n)


def add_volatility(px: pd.DataFrame) -> pd.DataFrame:
    """Annualized close-to-close std of log returns and Garman-Klass volatility."""
    g = px["permno"]
    lr = np.log1p(px["ret"])
    ok = (px["dlyopen"] > 0) & (px["dlyhigh"] > 0) & (px["dlylow"] > 0) & (px["prc"] > 0)
    gk = (0.5 * np.log(px["dlyhigh"] / px["dlylow"]) ** 2
          - (2 * np.log(2) - 1) * np.log(px["prc"] / px["dlyopen"]) ** 2).clip(lower=0).where(ok)
    for label, n in VOL_WINDOWS.items():
        px[f"vol_gk_{label}"] = np.sqrt(_rolling(gk, g, n, False) * TRADING_DAYS_PER_YEAR)
        px[f"vol_close_{label}"] = _rolling(lr, g, n, True) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return px


# ---------- Monthly table and calendar ----------

def build_calendar(px: pd.DataFrame) -> pd.DataFrame:
    real = px[~px["is_delisting_row"]]
    cal = real.groupby(real["date"].dt.to_period("M"))["date"].agg(month_end="max", trading_days="nunique")
    return cal.reset_index(names="month")


def build_monthly(px: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """One row per permno and month with any row: ret_m compounds every row (delisting row included); prc, shrout,
    mkt cap, momentum and volatility come from the last real trading day; siccd is the last non-zero SIC so far."""
    month = px["date"].dt.to_period("M")
    keys = [px["permno"], month]
    lr = np.log1p(px["ret"])
    sic = px["siccd"].where(px["siccd"] > 0).groupby(px["permno"]).ffill()
    out = pd.DataFrame({
        "ret_m": np.expm1(lr.groupby(keys, sort=False).sum(min_count=1)),
        "trading_days": (~px["is_delisting_row"]).groupby(keys, sort=False).sum().astype("int16"),
        "delisted_in_month": px["is_delisting_row"].groupby(keys, sort=False).any(),
        "delist_ret_filled": px["delist_ret_filled"].groupby(keys, sort=False).any(),
        "permco": px["permco"].groupby(keys, sort=False).last(),
        "ticker": px["ticker"].astype(object).groupby(keys, sort=False).last(),
        "siccd": sic.groupby(keys, sort=False).last(),
    })
    out.index.names = ["permno", "month"]
    out = out.reset_index()
    rows = px[month_end_rows(px)]
    end = rows[["permno", "date", "prc", "shrout", "mkt_cap_musd", *MOM_COLS, *VOL_COLS]].rename(
        columns={"date": "last_trade_date"})
    end.insert(1, "month", rows["date"].dt.to_period("M"))
    out = out.merge(end, on=["permno", "month"], how="left")
    out = out.merge(calendar[["month", "month_end"]], on="month", how="left")
    cols = ["month_end", "permno", "permco", "ticker", "siccd", "last_trade_date", "prc", "shrout", "mkt_cap_musd",
            *MOM_COLS, *VOL_COLS, "ret_m", "trading_days", "delisted_in_month", "delist_ret_filled"]
    return out[cols].sort_values(["month_end", "permno"]).reset_index(drop=True)


def build_daily(px: pd.DataFrame) -> pd.DataFrame:
    px["cfacpr"] = px["dlycumfacpr"]
    px["prc_adj"] = (px["prc"] / px["cfacpr"].where(px["cfacpr"] > 0)).fillna(px["prc"])
    px["vol"] = px["dlyvol"].where(px["dlyvol"] >= 0)
    return px[list(DAILY_COLUMNS)]


def write_daily(daily: pd.DataFrame, out_dir) -> dict:
    """One Parquet file per year (the layout of the repo's daily data), sorted by permno and date."""
    years = {}
    for year, part in daily.groupby(daily["date"].dt.year, sort=True):
        path = out_dir / "daily" / f"year={year}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part = part.assign(ticker=part["ticker"].astype(object))
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), path, compression=PARQUET_COMPRESSION,
                       row_group_size=250_000)
        years[str(year)] = len(part)
    return years


def read_delistings(path) -> pd.DataFrame:
    dl = pd.read_csv(path, parse_dates=["delistingdt"], dtype={"delactiontype": str, "delreasontype": str,
                                                                "delpaymenttype": str})
    dl.columns = dl.columns.str.lower()
    dl["permno"] = dl["permno"].astype("int32")
    return dl


def build_delistings(dl: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    flag = px[px["is_delisting_row"]]
    flag = pd.DataFrame({"permno": flag["permno"], "flagged_row_date": flag["date"],
                         "flagged_row_prc": flag["dlyprc"].abs().where(lambda s: s > 0), "delret_used": flag["ret"]})
    out = dl.rename(columns={"delistingdt": "delisting_date"}).merge(flag, on="permno", how="left")
    return out[["permno", "delisting_date", "flagged_row_date", "flagged_row_prc", "delactiontype", "delreasontype",
                "delpaymenttype", "delret", "delret_used"]].sort_values(["delisting_date", "permno"])


# ---------- Checks ----------

def run_checks(px, monthly, calendar, dl, n_dups, n_conflicts, listed, seed=7) -> list[Check]:
    checks = [check("conflicting duplicate rows", n_conflicts == 0, f"{n_conflicts} permno-dates differ"),
              check("unique permno and date", not px.duplicated(["permno", "date"]).any()),
              vs_last_run("exact duplicate rows removed", n_dups, "price_duplicates")]

    flags = px[px["is_delisting_row"]]
    per = flags.groupby("permno").size()
    last_is_flag = px.groupby("permno")["is_delisting_row"].last()
    checks.append(check("one delisting row per delisted permno, and it is the last row",
                        bool((per <= 1).all() and last_is_flag[per.index].all()),
                        f"{len(flags):,} delisting rows, {int(per.gt(1).sum())} permnos with more than one"))
    checks.append(check("delisting rows = delistings.csv", set(flags["permno"]) == set(dl["permno"]),
                        f"{len(flags):,} rows vs {len(dl):,} delistings"))
    per_m = monthly[monthly["delisted_in_month"]].groupby("permno").size()
    checks.append(check("every delisting falls in exactly one month",
                        bool(len(per_m) == len(dl) and (per_m == 1).all()), f"{len(per_m):,} permnos"))
    n_fill, n_blank = int(px["delist_ret_filled"].sum()), int(dl["delret"].isna().sum())
    checks.append(check("filled delisting returns = blank delret in delistings.csv", n_fill == n_blank,
                        f"{n_fill} filled, {n_blank} blank"))
    checks.append(vs_last_run("delisting returns filled", n_fill, "delisting_fills"))
    both = flags.merge(dl[["permno", "delret"]], on="permno").dropna(subset=["dlyret", "delret"])
    same = float(np.isclose(both["dlyret"], both["delret"], atol=1e-6).mean()) if len(both) else 1.0
    checks.append(check("delisting row return equals delistings.csv delret", same >= 0.99,
                        f"{same:.2%} of {len(both):,}", level=WARN))

    rng = np.random.default_rng(seed)
    pick = rng.choice(px["permno"].unique(), min(25, px["permno"].nunique()), replace=False)
    sub = px[px["permno"].isin(pick)]
    comp = (1 + sub["ret"]).groupby([sub["permno"], sub["date"].dt.to_period("M")]).prod(min_count=1) - 1
    comp = comp.rename("from_daily").reset_index().rename(columns={"date": "month"})
    mine = monthly[monthly["permno"].isin(pick)].assign(month=lambda d: d["month_end"].dt.to_period("M"))
    m = mine.merge(comp, on=["permno", "month"])
    checks.append(check("ret_m equals the compounded daily returns (25 random permnos)",
                        len(m) == len(mine) and bool(np.allclose(m["ret_m"], m["from_daily"], rtol=1e-9, atol=1e-12,
                                                                 equal_nan=True)), f"{len(m):,} permno-months"))

    used = monthly.dropna(subset=["last_trade_date"]).merge(
        px[["permno", "date", "is_delisting_row"]], left_on=["permno", "last_trade_date"],
        right_on=["permno", "date"], how="left")
    only_flag = monthly["trading_days"] == 0
    checks.append(check("no month-end price or market cap from a delisting row",
                        bool(not used["is_delisting_row"].any()
                             and monthly.loc[only_flag, ["prc", "mkt_cap_musd"]].isna().all().all()),
                        f"{int(only_flag.sum()):,} months hold only a delisting row"))

    big_ret, big_prc = px[px["ret"] > 10], px[px["prc"] > 100_000]
    detail = f"{len(big_ret)} returns, {len(big_prc)} prices"
    if len(big_ret):
        top = big_ret.nlargest(3, "ret")
        detail += "; largest: " + ", ".join(f"permno {r.permno} {r.date:%Y-%m-%d} {r.ret:+.0%}"
                                            for r in top.itertuples())
    checks.append(check("daily returns above +1000% or prices above $100,000", big_ret.empty and big_prc.empty,
                        detail, level=WARN))
    if listed is not None:
        have = set(px["permno"].tolist())
        missing = sorted(listed - have)
        checks.append(check("permnos in permno_list.txt have CIZ rows", not missing,
                            f"{len(missing)} without rows: {missing[:10]}; {len(have - listed)} not listed",
                            level=WARN))
    months = calendar["month"]
    checks.append(check("calendar has every month", len(months) == (months.max() - months.min()).n + 1,
                        f"{months.min()} to {months.max()}"))
    checks.append(vs_last_run("daily rows", len(px), "daily_rows"))
    checks.append(vs_last_run("monthly rows", len(monthly), "monthly_rows"))
    return checks


# ---------- Stage ----------

def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    t0 = time.time()
    px = read_ciz(paths.raw / CIZ_FILE)
    n_raw = len(px)
    px, n_dups, n_conflicts = drop_duplicates(px)
    dl = read_delistings(paths.raw / "delistings.csv")
    px = add_volatility(add_momentum(fill_delisting_returns(clean_prices(px), dl)))
    print(f"  read {n_raw:,} rows, {n_dups:,} duplicates dropped, cleaned in {time.time() - t0:.0f}s", flush=True)

    calendar = build_calendar(px)
    monthly = build_monthly(px, calendar)
    delistings = build_delistings(dl, px)
    listed_path = paths.query_inputs / "permno_list.txt"
    listed = set(int(x) for x in listed_path.read_text().split()) if listed_path.exists() else None
    checks = run_checks(px, monthly, calendar, dl, n_dups, n_conflicts, listed)

    daily = build_daily(px)
    years = write_daily(daily, out)
    write_parquet(monthly, out / "monthly.parquet")
    write_parquet(calendar.assign(month=calendar["month"].astype(str)), out / "calendar.parquet")
    write_parquet(delistings, out / "delistings.parquet")

    fills = px[px["delist_ret_filled"]]
    tables = {
        "daily/": {**table_info(daily, "date", ["ret", "prc", "mkt_cap_musd", "vol", "mom_1m", "mom_12m",
                                                "vol_close_12m", "vol_gk_12m"]), "years": years},
        "monthly.parquet": table_info(monthly, "month_end", ["ret_m", "prc", "mkt_cap_musd", "mom_12m",
                                                             "vol_close_12m", "trading_days"]),
        "calendar.parquet": table_info(calendar, "month_end", ["trading_days"]),
        "delistings.parquet": table_info(delistings, "delisting_date", ["delret", "delret_used",
                                                                        "flagged_row_prc"]),
    }
    extra = {"raw_rows": n_raw, "duplicates_removed": n_dups,
             "delisting_fills": {"total": len(fills), "gdr_minus_30pct": int((fills["ret"] == -0.30).sum()),
                                 "zero": int((fills["ret"] == 0).sum())}}
    summary = write_summary(out, STAGE, tables, checks, extra)
    print(f"  daily {len(daily):,} rows in {len(years)} years | monthly {len(monthly):,} rows | "
          f"{len(dl):,} delistings, {len(fills)} returns filled | {time.time() - t0:.0f}s")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
