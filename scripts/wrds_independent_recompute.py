#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/wrds_independent_recompute.py
PURPOSE: Recomputes a sample of factor-panel values straight from the raw WRDS CSVs, as an independent check of
         the pipeline (stage s09_validate). It deliberately does NOT import pipeline code: every rule is written
         out again here from pipeline/PLAN.md.

Usage:
    python scripts/wrds_independent_recompute.py RAW_DIR SAMPLE_CSV OUT_CSV

SAMPLE_CSV has columns permno, date (a month end t). OUT_CSV gets per row: b_p, roa, mom_12_1, e_p, ev_ebitda,
ret_m (the return of month t) and what was used (fund_period_end, company_mkt_cap_musd, compustat_flag).

Rules re-implemented (PLAN.md s02, s03, s07):
    prices: |DlyPrc| > 0 on real trading rows (DlyDelFlg != Y); returns below -100% are missing; a delisting
            row without a return gets -30% for delisting action GDR, else 0
    month end: the stock's last real trading row in the month; momentum = compounded return over the last 21 / 252
            rows (delisting rows included), needing 80% of them; 12-1 = (1 + m12) / (1 + m1) - 1
    company market cap: all permnos in the CIZ file with the same permco at their last real row in the month; plus
            Compustat shares (cshoq x cfacshr at the quarter end / cfacshr at t) at the largest class's price when
            they exceed the CRSP shares by > 5%, no class is an ADR, and the excess does not last <= 3 months
            within 15% (computed here over calendar months)
    fundamentals: one row per (permno, fyearq, fqtr), the most complete, then the latest rdq; report date = rdq or
            datadate + 45 days (+ 90 for Q4); drop quarters filed after a newer one; latest report date <= t (ties:
            the latest period end, then fiscal year and quarter); stale after 365 days; TTM = 4 consecutive
            fiscal quarters; book equity = seqq (else ceqq + pstkq, else atq - ltq) + txditcq - pstkq
    factors: B/P = book equity / company cap (book > 0); ROA = TTM niq / atq; E/P = TTM ibq / company cap;
            EV/EBITDA = (company cap + debt - cash + preferred + minority) / TTM oibdpq (EV > 0, EBITDA >= $1M);
            price ratios only for USD statements and prices >= $0.01
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


def read_ciz(raw: Path, permcos: set) -> pd.DataFrame:
    cols = ["permno", "PERMCO", "DlyCalDt", "DlyDelFlg", "DlyPrc", "DlyRet", "ShrOut", "DlyCumFacShr", "SecurityNm"]
    t = pacsv.read_csv(raw / "r2000_price_crsp_ciz.csv", convert_options=pacsv.ConvertOptions(
        include_columns=cols, column_types={"SecurityNm": pa.string(), "DlyDelFlg": pa.string()}))
    t = t.filter(pc.is_in(t["PERMCO"], value_set=pa.array(sorted(permcos), type=t["PERMCO"].type)))
    df = t.to_pandas()
    df.columns = ["permno", "permco", "date", "delflg", "prc", "ret", "shrout", "cfacshr", "name"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["permno", "date"], keep="first").sort_values(["permno", "date"]).reset_index(drop=True)
    df["dl"] = df["delflg"].eq("Y")
    df["prc"] = df["prc"].abs().where(lambda s: s > 0).where(~df["dl"])
    df["shrout"] = df["shrout"].where(df["shrout"] > 0)
    df["ret"] = df["ret"].where(df["ret"] >= -1)
    dl = pd.read_csv(raw / "delistings.csv", dtype={"delactiontype": str})
    action = df["permno"].map(dl.drop_duplicates("permno").set_index("permno")["delactiontype"])
    fill = df["dl"] & df["ret"].isna()
    df.loc[fill, "ret"] = np.where(action[fill].eq("GDR"), -0.30, 0.0)
    df["adr"] = df["name"].str.split(";").str[1].str.split().apply(lambda w: isinstance(w, list) and "ADR" in w)
    df["month"] = df["date"].dt.to_period("M")
    return df


def momentum(g: pd.DataFrame, end_idx: int, n: int) -> float:
    pos = g.index.get_loc(end_idx)
    if pos + 1 < n:
        return np.nan
    w = g["ret"].iloc[pos + 1 - n: pos + 1]
    if w.notna().sum() < 0.8 * n:
        return np.nan
    return float(np.expm1(np.log1p(w.dropna()).sum()))


def read_fundamentals(raw: Path, permnos: set) -> pd.DataFrame:
    fq = pd.read_csv(raw / "fundamentals_raw.csv", dtype={"GVKEY": str, "curcdq": str}, low_memory=False)
    fq.columns = fq.columns.str.lower()
    fq = fq[fq["lpermno"].isin(permnos)].copy()
    fq["datadate"], fq["rdq"] = pd.to_datetime(fq["datadate"]), pd.to_datetime(fq["rdq"], errors="coerce")
    fq = fq.dropna(subset=["lpermno", "datadate", "fyearq", "fqtr"])
    fq["q"] = fq["fyearq"].astype(int) * 4 + fq["fqtr"].astype(int)
    fq["filled"] = fq.notna().sum(axis=1)
    fq = fq.sort_values(["lpermno", "q", "filled", "rdq"], na_position="first").drop_duplicates(
        ["lpermno", "q"], keep="last")
    fq["report_date"] = fq["rdq"].fillna(fq["datadate"] + pd.to_timedelta(np.where(fq["fqtr"] == 4, 90, 45), "D"))
    by_q = fq.set_index(["lpermno", "q"])

    def ttm(item):
        vals = [by_q[item].reindex(pd.MultiIndex.from_arrays([fq["lpermno"], fq["q"] - k])).to_numpy()
                for k in range(4)]
        return vals[0] + vals[1] + vals[2] + vals[3]
    fq["ni_ttm"], fq["ib_ttm"], fq["ebitda_ttm"] = ttm("niq"), ttm("ibq"), ttm("oibdpq")
    pref = fq["pstkq"].fillna(0)
    fq["book"] = fq["seqq"].fillna(fq["ceqq"] + pref).fillna(fq["atq"] - fq["ltq"]) + fq["txditcq"].fillna(0) - pref
    fq["claims"] = (fq["dlttq"].fillna(0) + fq["dlcq"].fillna(0) - fq["cheq"].fillna(0) + pref
                    + fq["mibtq"].fillna(0))
    fq = fq.sort_values(["lpermno", "report_date", "datadate", "fyearq", "fqtr"])
    fq = fq[fq["datadate"] >= fq.groupby("lpermno")["datadate"].cummax()]          # late-filed older quarters
    return fq


def fundamentals_at(fq: pd.DataFrame, permno: int, t: pd.Timestamp):
    f = fq[(fq["lpermno"] == permno) & (fq["report_date"] <= t)]
    if f.empty:
        return None
    r = f.iloc[-1]
    return None if (t - r["datadate"]).days > 365 else r


def month_end_rows(df: pd.DataFrame) -> pd.DataFrame:
    real = df[~df["dl"]]
    return real.groupby(["permno", "month"]).tail(1).set_index(["permno", "month"])


def company_shares(df_end, permco, month):
    rows = df_end[(df_end["permco"] == permco) & (df_end.index.get_level_values("month") == month)]
    return rows


def compustat_flag(df_end, fq, permno, permco, month, t):
    """(flag, excess shares in millions, main price) for one company-month, before the episode rule."""
    rows = company_shares(df_end, permco, month)
    if rows.empty or rows["adr"].any():
        return False, np.nan, np.nan, np.nan
    f = fundamentals_at(fq, permno, t)
    if f is None or not f["cshoq"] > 0:
        return False, np.nan, np.nan, np.nan
    try:
        cf_qe = df_end.loc[(permno, f["datadate"].to_period("M")), "cfacshr"]
        cf_t = df_end.loc[(permno, month), "cfacshr"]
    except KeyError:
        return False, np.nan, np.nan, np.nan
    cs = f["cshoq"] * cf_qe / cf_t
    crsp = rows["shrout"].sum(min_count=1) / 1000
    caps = (rows["prc"] * rows["shrout"]).dropna()
    main = rows.loc[caps.idxmax(), "prc"] if len(caps) else np.nan
    return bool(cs - crsp > 0.05 * crsp and pd.notna(main)), cs - crsp, main, cs / crsp


def main(raw_dir: str, sample_csv: str, out_csv: str) -> None:
    raw = Path(raw_dir)
    sample = pd.read_csv(sample_csv, parse_dates=["date"])
    permcos_of = pacsv.read_csv(raw / "r2000_price_crsp_ciz.csv", convert_options=pacsv.ConvertOptions(
        include_columns=["permno", "PERMCO"])).to_pandas().drop_duplicates()
    wanted = set(permcos_of.loc[permcos_of["permno"].isin(sample["permno"]), "PERMCO"])
    df = read_ciz(raw, wanted)
    fq = read_fundamentals(raw, set(df["permno"]))
    end = month_end_rows(df)
    out = []
    for s in sample.itertuples():
        t, month = s.date, s.date.to_period("M")
        g = df[df["permno"] == s.permno]
        rec = {"permno": s.permno, "date": t.date()}
        inm = g[g["month"] == month]
        rec["ret_m"] = float(np.expm1(np.log1p(inm["ret"]).sum(min_count=1))) if inm["ret"].notna().any() else np.nan
        if (s.permno, month) not in end.index:
            out.append(rec)
            continue
        me = end.loc[(s.permno, month)]
        idx = g.index[(g["date"] == me["date"])][0]
        m1, m12 = momentum(g, idx, 21), momentum(g, idx, 252)
        rec["mom_12_1"] = (1 + m12) / (1 + m1) - 1
        permco = me["permco"]
        rows = company_shares(end, permco, month)
        cap = (rows["prc"] * rows["shrout"]).sum(min_count=1) / 1000
        flag, extra, main_prc, ratio = compustat_flag(end, fq, s.permno, permco, month, t)
        if flag and ratio <= 1.15:                    # episode rule: is the flag a short one?
            run, k = 1, 1
            for step in (-1, 1):
                k = 1
                while True:
                    m2 = month + step * k
                    t2 = end[end.index.get_level_values("month") == m2]["date"].max()
                    if pd.isna(t2):
                        break
                    f2, _, _, r2 = compustat_flag(end, fq, s.permno, permco, m2, t2)
                    if not f2:
                        break
                    run += 1
                    ratio = max(ratio, r2)
                    k += 1
            if run <= 3 and ratio <= 1.15:
                flag = False
        if flag:
            cap = cap + extra * main_prc
        rec.update({"company_mkt_cap_musd": cap, "compustat_flag": flag})
        f = fundamentals_at(fq, s.permno, t)
        if f is not None:
            rec["fund_period_end"] = f["datadate"].date()
            rec["roa"] = f["ni_ttm"] / f["atq"] if f["atq"] else np.nan
            if f["curcdq"] == "USD" and me["prc"] >= 0.01 and pd.notna(cap):
                rec["b_p"] = f["book"] / cap if f["book"] > 0 else np.nan
                rec["e_p"] = f["ib_ttm"] / cap
                ev = cap + f["claims"]
                rec["ev_ebitda"] = ev / f["ebitda_ttm"] if ev > 0 and f["ebitda_ttm"] >= 1 else np.nan
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
