"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s01_membership.py
PURPOSE: Stage s01_membership: monthly index snapshots from three sources (the client's 9/30 lists before
         2008-02, IWM holdings, and the list members IWM's old fund id hides until 2010-05), membership periods,
         permno fill from CUSIP, known gaps and their fixes, and the non-member (merger) stocks excluded from the
         universe.

In:  raw/iwm_holdings_raw.csv, raw/r2000_price_crsp_ciz.csv (permno, date, flag, price, shares: month ends, first
     trading days and month-end market caps for rule 6, computed by s02's own function), raw/crsp_names_history.csv
     (trading on a date, names), processed/s00c_client_map/client_permno_map.parquet (the client lists' permnos),
     the permno upload list (only to compare)
Out: processed/s01_membership/snapshots.parquet             one row per report_dt and permno, with membership_source
     processed/s01_membership/periods.parquet               one row per membership period (one source each)
     processed/s01_membership/excluded_non_members.parquet  stock-snapshots excluded from the universe
     processed/s01_membership/known_gaps.parquet, known_gaps.md   IWM holdings still without a permno
     processed/s01_membership/gap_fixes.parquet             the known-gap CUSIPs and the permno each one gets
     processed/s01_membership/permno_list.txt               the upload list for the permno-based WRDS queries
     processed/s01_membership/summary.json

Rules (pipeline/PLAN.md, s01_membership):
    1. a line without a permno gets the most common permno of the same CUSIP on other dates (audit A8)
    2. lines still without a permno that look like common stock are the known gaps (A8)
    3. one row per stock per snapshot: weights summed, earliest eff_dt, names from the largest line
    4. a stock missing from 2+ consecutive snapshots starts a new period; one missing snapshot is bridged (A7)
    5. permno_list.txt = every snapshot permno
    6. a stock-snapshot is out of the universe if (a) its month-end market cap is > 3x the largest other stock in
       the snapshot and its weight is below 0.05%, or (b) the stock is in <= 3 snapshots in total, all with weight
       0.00%, and its market cap is above the snapshot's 95th percentile (decision 11, revised)
    7. before 2008-02: one snapshot per month end from the client's 9/30 lists (decision 33a)
    8. 2008-02 to 2010-05: list members missing from the IWM snapshot are added (client_fill, decision 33b)
    9. list timing: the 9/30/Y list counts from the first month end after year Y's reconstitution until the next
       list does; a member first trading after that reconstitution counts from the September Y month end; a member
       counts only while trading
   10. periods never mix sources; rule 6 runs on every source (decision 33d)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.checks import PASS, WARN, Check, check, vs_last_run
from pipeline.config import (
    CLIENT_FILL_LAST_MONTH,
    CLIENT_LIST_FIRST_MONTH,
    GAP_FIX_CLIENT_WINDOW_MONTHS,
    GAP_FIX_MIN_SHARES,
    IWM_FIRST_MONTH,
    KNOWN_MISSING_MONTHS,
    MEMBERSHIP_BRIDGE_SNAPSHOTS,
    MEMBERSHIP_SOURCES,
    NAMES_FILE,
    NEIGHBOUR_MISSING_MAX,
    NON_MEMBER_CAP_MULTIPLE,
    NON_MEMBER_CAP_QUANTILE_B,
    NON_MEMBER_MAX_SNAPSHOTS,
    NON_MEMBER_MAX_WEIGHT_A,
    NON_STOCK_PATTERN,
    SNAPSHOT_SIZE_RANGE,
    SNAPSHOT_WEIGHT_RANGE,
    DataPaths,
    client_list_year,
    reconstitution_date,
)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages import s00c_client_map, s02_prices

STAGE = "s01_membership"
HOLDINGS_FILE = "iwm_holdings_raw.csv"
SNAPSHOT_ID_COLUMNS = ["report_dt", "eff_dt", "permno", "weight", "ticker", "security_name", "cusip", "n_lines",
                       "permno_from_cusip", "permno_from_gap_fix"]


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
    if "permno_from_gap_fix" not in h.columns:
        h = h.assign(permno_from_gap_fix=False)
    keys = ["report_dt", "permno"]
    agg = h.groupby(keys).agg(eff_dt=("eff_dt", "min"), weight=("percent_tna", "sum"),
                              n_lines=("percent_tna", "size"), permno_from_cusip=("permno_from_cusip", "any"),
                              permno_from_gap_fix=("permno_from_gap_fix", "any"))
    names = (h.sort_values(keys + ["percent_tna", "nbr_shares", "line"], ascending=[True, True, False, False, True])
              .groupby(keys)[["ticker", "security_name", "cusip"]].first())
    snap = agg.join(names).reset_index()
    return snap[SNAPSHOT_ID_COLUMNS].sort_values(keys).reset_index(drop=True)


def _ids(values) -> str:
    return " ".join(map(str, sorted({int(v) for v in values})))


def gap_fixes(gaps: pd.DataFrame, h: pd.DataFrame, names: s00c_client_map.Names, cmap: pd.DataFrame
              ) -> pd.DataFrame:
    """Rule 2b: a permno for each known-gap CUSIP where one is certain. The IWM file labels old holdings with the
    security's current CUSIP, so CRSP's CUSIP -> permno counts when the permno traded at any time during the gap's
    IWM dates (a permno still trading at the end of the names file counts for IWM dates after it). Otherwise a
    client-list row with that CUSIP, or with the ticker and a matching name, whose 9/30 is within
    GAP_FIX_CLIENT_WINDOW_MONTHS of the IWM dates. A gap whose lines all hold fewer than GAP_FIX_MIN_SHARES shares
    is a stray line and stays a gap."""
    out = gaps.copy()
    if gaps.empty:
        return out.assign(crsp_permnos_valid="", crsp_permnos_any="", client_permnos="", client_via="",
                          fix_permno=pd.array([], dtype="Int64"), fix_source="")
    n = names.n
    names_end = n["secinfoenddt"].max()
    life = n.groupby("permno").agg(p_from=("secinfostartdt", "min"), p_to=("secinfoenddt", "max"))
    m = gaps[["cusip", "first", "last"]].merge(n[["cusip", "permno"]].drop_duplicates(), on="cusip").join(
        life, on="permno")
    m["after_end"] = (m["p_to"] == names_end) & (m["first"] > names_end)
    m["overlap"] = (m[["p_to", "last"]].min(axis=1) - m[["p_from", "first"]].max(axis=1)).dt.days
    m.loc[m["after_end"], "overlap"] = 0
    valid = m[m["overlap"] >= 0].groupby("cusip")["permno"].agg(lambda x: sorted(set(x)))
    anyp = m.groupby("cusip")["permno"].agg(lambda x: sorted(set(x)))
    after_end = m.groupby("cusip")["after_end"].any()

    cm = cmap[cmap["final_permno"].notna()]
    by_cusip = gaps.merge(cm[["cusip8", "date", "final_permno"]], left_on="cusip", right_on="cusip8")
    by_tkr = gaps.merge(cm[["ticker", "date", "final_permno", "name"]], on="ticker")
    by_tkr = by_tkr[[s00c_client_map.name_ok(a, b) for a, b in zip(by_tkr["security_name"], by_tkr["name"])]]
    hits = pd.concat([by_cusip.assign(via="cusip"), by_tkr.assign(via="ticker+name")], ignore_index=True)
    win = pd.DateOffset(months=GAP_FIX_CLIENT_WINDOW_MONTHS)
    hits = hits[(hits["date"] >= hits["first"] - win) & (hits["date"] <= hits["last"] + win)]
    client = hits.groupby("cusip").agg(permnos=("final_permno", lambda x: sorted({int(v) for v in x})),
                                       via=("via", lambda x: "+".join(sorted(set(x)))))
    shares = h[h["permno"].isna() & h["cusip"].isin(gaps["cusip"])].groupby("cusip")["nbr_shares"].max()

    def decide(cusip):
        v, cl = valid.get(cusip, []), client["permnos"].get(cusip, [])
        if shares.get(cusip, np.inf) < GAP_FIX_MIN_SHARES:
            return None, f"stray line: fewer than {GAP_FIX_MIN_SHARES} shares"
        if len(v) == 1:
            src = "crsp_cusip" + (" (IWM dates after names file end)" if after_end.get(cusip, False) else "")
            return v[0], src + ("" if not cl or cl == v else " (client differs)")
        if len(v) > 1:
            return None, "crsp_cusip: several permnos, review"
        if len(cl) == 1:
            return cl[0], f"client_map ({client['via'][cusip]})"
        if cusip in anyp.index:
            return None, "CUSIP in CRSP, permno not trading on the IWM dates"
        return None, "no fix"
    dec = [decide(c) for c in out["cusip"]]
    out["crsp_permnos_valid"] = out["cusip"].map(lambda c: _ids(valid.get(c, [])))
    out["crsp_permnos_any"] = out["cusip"].map(lambda c: _ids(anyp.get(c, [])))
    out["client_permnos"] = out["cusip"].map(lambda c: _ids(client["permnos"].get(c, [])))
    out["client_via"] = out["cusip"].map(client["via"]).fillna("")
    out["fix_permno"] = pd.array([d[0] for d in dec], dtype="Int64")
    out["fix_source"] = [d[1] for d in dec]
    return out


def apply_gap_fixes(h: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
    """Holdings lines without a permno whose CUSIP has a fix get that permno (permno_from_gap_fix)."""
    fx = fixes.dropna(subset=["fix_permno"]).set_index("cusip")["fix_permno"].astype("float64")
    sel = h["permno"].isna() & h["cusip"].isin(fx.index)
    h = h.copy()
    h.loc[sel, "permno"] = h.loc[sel, "cusip"].map(fx)
    h["permno_from_gap_fix"] = sel
    return h


def names_on(rows: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """CRSP ticker, name and CUSIP of each (permno, report_dt): the latest names row starting on or before it."""
    left = rows.assign(_i=np.arange(len(rows)), permno=rows["permno"].astype("int64")).sort_values("report_dt")
    right = (names[["permno", "secinfostartdt", "ticker", "issuernm", "cusip"]].dropna(subset=["secinfostartdt"])
             .sort_values("secinfostartdt"))
    m = pd.merge_asof(left, right, left_on="report_dt", right_on="secinfostartdt", by="permno", direction="backward")
    m = m.sort_values("_i")
    return pd.DataFrame({"ticker": m["ticker"].to_numpy(), "security_name": m["issuernm"].to_numpy(),
                         "cusip": m["cusip"].to_numpy()}, index=rows.index)


class ClientLists:
    """The client lists' members (s00c final permnos) and the timing rules (9) that decide who counts at a date.
    month_ends (month -> last CRSP trading day) decides the list in force; without it the date itself does."""

    def __init__(self, cmap: pd.DataFrame, names: s00c_client_map.Names, first_trade: pd.Series,
                 month_ends: pd.Series | None = None):
        m = cmap[cmap["final_permno"].notna()]
        self.members = pd.DataFrame({"list_year": m["year"].astype("int32").to_numpy(),
                                     "permno": m["final_permno"].astype("int32").to_numpy(),
                                     "weight": m["weight"].astype("float64").to_numpy()})
        self.names, self.first_trade = names, first_trade
        self.month_ends = month_ends if month_ends is not None else pd.Series(dtype="datetime64[ns]")

    def at(self, date: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
        """Members counting at a month end: the list in force, trading on the date, and an IPO after that year's
        reconstitution only from the September month end."""
        y = client_list_year(self.month_ends.get(date.to_period("M"), date))
        mem = self.members[self.members["list_year"] == y]
        trading = mem["permno"].isin(self.names.live(date))
        first = mem["permno"].map(self.first_trade)
        ipo = first > pd.Timestamp(reconstitution_date(y))
        held_back = ipo & (date < pd.Timestamp(year=y, month=9, day=1))
        keep = trading & ~held_back
        return mem[keep], {"list_year": int(y), "list_members": len(mem), "not_trading": int((~trading).sum()),
                           "ipo_after_reconstitution": int(ipo.sum()),
                           "ipo_held_back": int((held_back & trading).sum()),
                           "no_crsp_prices": int(first.isna().sum())}


def client_list_snapshots(lists: ClientLists, month_ends: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """Rule 7: one snapshot per month end CLIENT_LIST_FIRST_MONTH .. the month before IWM_FIRST_MONTH. report_dt
    = eff_dt = the month end (last CRSP trading day), so s07 uses it at that month end."""
    months = pd.period_range(CLIENT_LIST_FIRST_MONTH, pd.Period(IWM_FIRST_MONTH, "M") - 1, freq="M")
    parts, stats = [], []
    for mth in months:
        t = month_ends[mth]
        mem, st = lists.at(t)
        parts.append(mem.assign(report_dt=t, eff_dt=t))
        stats.append({"month": str(mth), **st})
    snap = pd.concat(parts, ignore_index=True)
    return snap.assign(membership_source="client_list"), stats


def client_fill(iwm: pd.DataFrame, lists: ClientLists) -> tuple[pd.DataFrame, list[dict]]:
    """Rule 8: IWM snapshots IWM_FIRST_MONTH..CLIENT_FILL_LAST_MONTH get the list members they miss, trading on
    the report date. The rows take the snapshot's latest eff_dt, so the snapshot becomes effective as before."""
    lo, hi = pd.Period(IWM_FIRST_MONTH, "M"), pd.Period(CLIENT_FILL_LAST_MONTH, "M")
    parts, stats = [], []
    for d, g in iwm.groupby("report_dt"):
        if not lo <= d.to_period("M") <= hi:
            continue
        mem, st = lists.at(d)
        add = mem[~mem["permno"].isin(set(g["permno"]))]
        parts.append(add.assign(report_dt=d, eff_dt=g["eff_dt"].max()))
        stats.append({"month": str(d.to_period("M")), "iwm": len(g), "fill": len(add), **st})
    empty = lists.members.iloc[:0].assign(report_dt=pd.Series(dtype="datetime64[ns]"),
                                          eff_dt=pd.Series(dtype="datetime64[ns]"))
    fill = pd.concat(parts, ignore_index=True) if parts else empty
    return fill.assign(membership_source="client_fill"), stats


def with_ids(rows: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Client rows in the snapshot layout: CRSP ticker, name and CUSIP on the date; one line each."""
    rows = rows.reset_index(drop=True)
    ids = names_on(rows, names)
    return rows.assign(permno=rows["permno"].astype("int32"), ticker=ids["ticker"],
                       security_name=ids["security_name"], cusip=ids["cusip"], n_lines=1, permno_from_cusip=False,
                       permno_from_gap_fix=False)


def apply_non_member_rules(snap: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    """Rule 6: adds mkt_cap_musd (month-end of report_dt), max_other_mkt_cap_musd, p95_mkt_cap_musd,
    n_snapshots_total, exclusion_rule ('', 'a', 'b', 'a+b') and in_universe."""
    snap = snap.assign(month=snap["report_dt"].dt.to_period("M"))
    snap = snap.merge(caps[["permno", "month", "mkt_cap_musd"]], on=["permno", "month"], how="left")
    mc = snap["mkt_cap_musd"]
    by = snap.groupby("report_dt")["mkt_cap_musd"]
    max1 = by.transform("max")
    max2 = by.transform(lambda s: s.nlargest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
    is_top = mc.eq(max1)
    unique_top = is_top & is_top.groupby(snap["report_dt"]).transform("sum").eq(1)
    snap["max_other_mkt_cap_musd"] = np.where(unique_top, max2, max1)
    snap["p95_mkt_cap_musd"] = by.transform(lambda s: s.quantile(NON_MEMBER_CAP_QUANTILE_B))
    # (a) far above every other stock, with an index weight that does not fit that size
    rule_a = ((mc > NON_MEMBER_CAP_MULTIPLE * snap["max_other_mkt_cap_musd"])
              & (snap["weight"] < NON_MEMBER_MAX_WEIGHT_A))

    # (b) briefly held at 0.00% and large for the index
    per = snap.groupby("permno").agg(n=("report_dt", "nunique"), w=("weight", lambda w: w.round(2).abs().max()))
    small = per.index[(per["n"] <= NON_MEMBER_MAX_SNAPSHOTS) & (per["w"] == 0)]
    rule_b = snap["permno"].isin(small) & (mc > snap["p95_mkt_cap_musd"])

    snap["n_snapshots_total"] = snap["permno"].map(per["n"]).astype("int32")
    snap["exclusion_rule"] = np.select([rule_a & rule_b, rule_a, rule_b], ["a+b", "a", "b"], default="")
    snap["in_universe"] = snap["exclusion_rule"].eq("")
    return snap.drop(columns="month")


def build_periods(snap: pd.DataFrame) -> pd.DataFrame:
    """Rules 4 and 10: membership periods over all snapshots, a new one when the source changes;
    excluded_non_member when every snapshot of the period is excluded from the universe."""
    if "membership_source" not in snap.columns:
        snap = snap.assign(membership_source="iwm")
    order = {d: i for i, d in enumerate(sorted(snap["report_dt"].unique()))}
    s = snap.assign(k=snap["report_dt"].map(order)).sort_values(["permno", "k"])
    gap = s.groupby("permno")["k"].diff().fillna(np.inf) > 1 + MEMBERSHIP_BRIDGE_SNAPSHOTS
    new_source = s.groupby("permno")["membership_source"].shift().ne(s["membership_source"])
    s["period"] = (gap | new_source).cumsum()                       # rule 10: one source per period
    s["excluded"] = ~s["in_universe"]
    p = s.groupby("period").agg(permno=("permno", "first"), index_effective=("report_dt", "min"),
                                index_through=("report_dt", "max"), first_known=("eff_dt", "min"),
                                n_snapshots=("report_dt", "size"), n_excluded=("excluded", "sum"),
                                membership_source=("membership_source", "first"),
                                ticker=("ticker", "last"), security_name=("security_name", "last"))
    p["excluded_non_member"] = p["n_excluded"].eq(p["n_snapshots"])
    cols = ["permno", "index_effective", "index_through", "first_known", "n_snapshots", "n_excluded",
            "excluded_non_member", "membership_source", "ticker", "security_name"]
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
        checks.append(check("permno_list.txt equals the permno upload list", listed == raw_list,
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


def membership_checks(snap: pd.DataFrame, cl_stats: list[dict], fill_other_class: int) -> list[Check]:
    """Rules 7-10: each source only in its months, a client_list snapshot every month before IWM."""
    src, m = snap["membership_source"], snap["report_dt"].dt.to_period("M")
    iwm_first, fill_last = pd.Period(IWM_FIRST_MONTH, "M"), pd.Period(CLIENT_FILL_LAST_MONTH, "M")
    cl_months = sorted(set(m[src.eq("client_list")]))
    expected = list(pd.period_range(CLIENT_LIST_FIRST_MONTH, iwm_first - 1, freq="M"))
    no_prices = sum(st["no_crsp_prices"] for st in cl_stats)
    return [
        check("membership_source is client_list, iwm or client_fill", bool(src.isin(MEMBERSHIP_SOURCES).all())),
        check(f"client_list only before {IWM_FIRST_MONTH}, iwm from it",
              bool((m[src.eq("client_list")] < iwm_first).all() and (m[src.eq("iwm")] >= iwm_first).all())),
        check(f"client_fill only {IWM_FIRST_MONTH} to {CLIENT_FILL_LAST_MONTH}",
              bool(m[src.eq("client_fill")].between(iwm_first, fill_last).all())),
        check(f"a client_list snapshot every month {CLIENT_LIST_FIRST_MONTH} to {iwm_first - 1}",
              cl_months == expected, f"{len(cl_months)} months"),
        check("client list members without CRSP prices", no_prices == 0, f"{no_prices} list-months", level=WARN),
        check("client_fill rows whose company is already in the IWM snapshot (another share class)",
              fill_other_class == 0, f"{fill_other_class} rows", level=WARN),
    ]


def membership_report(snap: pd.DataFrame, cmap: pd.DataFrame, cl_stats: list[dict], fl_stats: list[dict],
                      fixes: pd.DataFrame, h: pd.DataFrame) -> dict:
    """The numbers the summary keeps about the sources: stocks per month and source, the fill's share of market
    cap, list switches, the IPO rule, the gap fixes, and the overlap of IWM with the client lists."""
    u = snap[snap["in_universe"]]
    by = u.groupby(["report_dt", "membership_source"]).agg(n=("permno", "size"), cap=("mkt_cap_musd", "sum"))
    n = by["n"].unstack(fill_value=0).reindex(columns=list(MEMBERSHIP_SOURCES), fill_value=0)
    cap = by["cap"].unstack(fill_value=0.0).reindex(columns=list(MEMBERSHIP_SOURCES), fill_value=0.0)
    sets = u.groupby("report_dt")["permno"].apply(set)
    months = {}
    prev = None
    for d in n.index:
        cur = sets[d]
        months[str(d.date())] = {**{k: int(v) for k, v in n.loc[d].items()}, "in_universe": len(cur),
                                 "fill_cap_pct": round(100 * cap.loc[d, "client_fill"] / cap.loc[d].sum(), 3)
                                 if cap.loc[d].sum() else None,
                                 "entered": len(cur - prev) if prev is not None else None,
                                 "left": len(prev - cur) if prev is not None else None}
        prev = cur
    stats = {st["month"]: st for st in cl_stats + fl_stats}
    switches, prev_year = {}, None
    for d in months:                                   # the month ends where the list in force changes
        st = stats.get(d[:7])
        if st and st["list_year"] != prev_year:
            switches[d[:7]] = {"list_year": st["list_year"], **months[d]}
            prev_year = st["list_year"]
    ipo = {}
    for st in cl_stats + fl_stats:
        y = ipo.setdefault(st["list_year"], {"members": st["list_members"],
                                             "ipo_after_reconstitution": st["ipo_after_reconstitution"],
                                             "held_back_stock_months": 0})
        y["held_back_stock_months"] += st["ipo_held_back"]
    overlap = {}
    cm = cmap[cmap["final_permno"].notna()]
    iwm = u[u["membership_source"].eq("iwm")]
    for y in range(int(cm["year"].min()), int(cm["year"].max()) + 1):
        d = iwm.loc[iwm["report_dt"].dt.to_period("M") == pd.Period(f"{y}-09", "M"), "permno"]
        if d.empty:
            continue
        cl, iw = set(cm.loc[cm["year"] == y, "final_permno"].astype(int)), set(d)
        both = cl & iw
        overlap[y] = {"client": len(cl), "iwm_9_30": len(iw), "both": len(both),
                      "client_in_iwm_pct": round(100 * len(both) / len(cl), 2),
                      "iwm_in_client_pct": round(100 * len(both) / len(iw), 2)}
    fixed = fixes.dropna(subset=["fix_permno"])
    return {
        "rows_by_source": {k: int(v) for k, v in snap["membership_source"].value_counts().items()},
        "months": months, "list_switches": switches, "ipo_rule_by_list": ipo,
        "client_list_months": cl_stats, "client_fill_months": fl_stats,
        "gap_fixes": {"gaps_before": len(fixes), "fixed_cusips": len(fixed), "fixed_permnos":
                      int(fixed["fix_permno"].nunique()), "lines_fixed": int(h["permno_from_gap_fix"].sum()),
                      "sources": {k: int(v) for k, v in fixes["fix_source"].value_counts().items()}},
        "overlap_iwm_9_30_vs_client_list": overlap,
    }


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    names = s00c_client_map.Names(s00c_client_map.read_names(paths.raw / NAMES_FILE))
    cmap = pd.read_parquet(paths.stage_dir(s00c_client_map.STAGE) / s00c_client_map.OUTPUT)
    h, fill = fill_permno_from_cusip(load_holdings(paths.raw / HOLDINGS_FILE))
    fixes = gap_fixes(known_gaps(h), h, names, cmap)
    h = apply_gap_fixes(h, fixes)
    gaps = known_gaps(h)
    iwm = build_snapshots(h).assign(membership_source="iwm")

    px = s02_prices.read_ciz(paths.raw / s02_prices.CIZ_FILE, s02_prices.MARKET_CAP_COLUMNS)
    px, _, _ = s02_prices.drop_duplicates(px)
    px = s02_prices.clean_prices(px)
    caps = s02_prices.month_end_market_caps(px)
    month_ends = px.groupby(px["date"].dt.to_period("M"))["date"].max()
    first_trade = px.loc[~px["is_delisting_row"]].groupby("permno")["date"].min()
    del px

    lists = ClientLists(cmap, names, first_trade, month_ends)
    cl, cl_stats = client_list_snapshots(lists, month_ends)
    fl, fl_stats = client_fill(iwm, lists)
    snap = pd.concat([with_ids(cl, names.n), iwm, with_ids(fl, names.n)], ignore_index=True)
    snap["list_year"] = snap["list_year"].astype("Int32")
    snap = snap[SNAPSHOT_ID_COLUMNS + ["membership_source", "list_year"]].sort_values(
        ["report_dt", "permno"]).reset_index(drop=True)
    snap = apply_non_member_rules(snap, caps)
    periods = build_periods(snap)
    excluded = snap.loc[~snap["in_universe"], ["report_dt", "eff_dt", "permno", "ticker", "security_name",
                                                "mkt_cap_musd", "max_other_mkt_cap_musd", "p95_mkt_cap_musd",
                                                "weight", "n_snapshots_total", "exclusion_rule",
                                                "membership_source"]].rename(
        columns={"exclusion_rule": "rule"}).reset_index(drop=True)

    def money(v):
        return None if pd.isna(v) else round(float(v), 1)

    permco = names.n.sort_values("secinfoenddt").groupby("permno")["permco"].last()
    fl_rows = snap[snap["membership_source"].eq("client_fill")]
    iwm_permcos = snap[snap["membership_source"].eq("iwm")].assign(pc=lambda d: d["permno"].map(permco))
    iwm_permcos = iwm_permcos.groupby("report_dt")["pc"].apply(set)
    fill_other_class = sum(permco.get(p) in iwm_permcos.get(d, set())
                           for d, p in zip(fl_rows["report_dt"], fl_rows["permno"]))

    listed = set(snap["permno"].tolist())
    raw_list = set(int(x) for x in paths.permno_list.read_text().split()) if paths.permno_list.exists() else None
    checks = run_checks(snap, periods, h, fill, listed, raw_list) + membership_checks(snap, cl_stats,
                                                                                       fill_other_class)

    write_parquet(snap, out / "snapshots.parquet")
    write_parquet(periods, out / "periods.parquet")
    write_parquet(excluded, out / "excluded_non_members.parquet")
    write_parquet(gaps, out / "known_gaps.parquet")
    write_parquet(fixes, out / "gap_fixes.parquet")
    n_gap_dates = int(h.loc[h["permno"].isna() & h["cusip"].isin(gaps["cusip"]), "report_dt"].nunique())
    (out / "known_gaps.md").write_text(gaps_markdown(gaps, n_gap_dates), encoding="utf-8")
    (out / "permno_list.txt").write_text("\n".join(str(p) for p in sorted(listed)) + "\n")

    tables = {"snapshots.parquet": table_info(snap, "report_dt", ["weight", "mkt_cap_musd", "n_lines"]),
              "periods.parquet": table_info(periods, "index_effective", ["n_snapshots"]),
              "excluded_non_members.parquet": table_info(excluded, "report_dt", ["mkt_cap_musd", "weight"]),
              "known_gaps.parquet": table_info(gaps, "first", ["snapshots", "max_weight"]),
              "gap_fixes.parquet": {"rows": len(fixes), "fixed": int(fixes["fix_permno"].notna().sum())}}
    extra = {"in_universe_rows": int(snap["in_universe"].sum()),
             "cusip_fill": {k: v for k, v in fill.items() if k != "cusips_with_two_permnos"},
             "membership": membership_report(snap, cmap, cl_stats, fl_stats, fixes, h),
             "fill_rows_other_share_class": int(fill_other_class),
             "permno_list_vs_upload_list": None if raw_list is None else {
                 "listed": len(listed), "upload_list": len(raw_list), "new": sorted(listed - raw_list)[:50],
                 "n_new": len(listed - raw_list), "n_not_in_snapshots": len(raw_list - listed)},
             "excluded_non_members": [
                 {"security_name": r.security_name, "ticker": r.ticker, "permno": int(r.permno),
                  "report_dt": f"{r.report_dt:%Y-%m-%d}", "eff_dt": f"{r.eff_dt:%Y-%m-%d}",
                  "mkt_cap_musd": money(r.mkt_cap_musd), "max_other_mkt_cap_musd": money(r.max_other_mkt_cap_musd),
                  "p95_mkt_cap_musd": money(r.p95_mkt_cap_musd),
                  "weight_pct": round(float(r.weight), 4), "n_snapshots_total": int(r.n_snapshots_total),
                  "rule": r.rule, "membership_source": r.membership_source} for r in excluded.itertuples()]}
    summary = write_summary(out, STAGE, tables, checks, extra)
    rows = snap["membership_source"].value_counts()
    print(f"  {snap['report_dt'].nunique()} snapshots, {len(snap):,} rows ("
          + ", ".join(f"{k} {rows.get(k, 0):,}" for k in MEMBERSHIP_SOURCES) + f"), {len(listed):,} permnos, "
          f"{len(periods):,} periods | {fill['lines']:,} lines filled from CUSIP | "
          f"{int(fixes['fix_permno'].notna().sum())} gap CUSIPs fixed, {len(gaps)} known gaps left | "
          f"{len(excluded)} stock-snapshots excluded ({excluded['permno'].nunique()} stocks)")
    for c in checks:
        if c.status != PASS:
            print(f"  {c.status}  {c.name}: {c.detail}")
    return summary
