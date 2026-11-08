"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/stages/s00c_client_map.py
PURPOSE: Stage s00c_client_map: a CRSP permno for every row of the client's 9/30 Russell 2000 lists.

In:  processed/s00b_client/client_lists.parquet, raw/crsp_names_history.csv (CRSP security names with validity
     dates), raw/r2000_price_crsp_ciz.csv (the last September trading day only, streamed)
Out: processed/s00c_client_map/client_permno_map.parquet   one row per client row: status, candidates,
                                                            final_permno, final_method
     processed/s00c_client_map/known_gaps.parquet, known_gaps.md   client securities without a permno
     processed/s00c_client_map/summary.json

Rules (pipeline/PLAN.md, s00c_client_map; decision 34):
    1. CUSIP rows: the client CUSIP (8 characters) equals a CRSP CUSIP valid on the 9/30
    2. rows without a CUSIP ("#Calc"): the ticker valid on the 9/30, and the client name must match one of the
       permno's CRSP names (FactSet back-fills today's names, so every name the permno ever had counts)
    3. a CUSIP match whose company never had the client's name, while the client ticker + name point to another
       permno trading on the 9/30, takes that permno (cusip_corrected_by_ticker)
    4. fallback for rows still without a permno: the ticker (FactSet's .XXn suffix removed) on any date, a name
       match, the permno trading on the 9/30, a single candidate, and a CRSP price on the last September trading
       day (ticker_any_date, cusip_row_ticker_any_date); without the price the fallback is rejected
    5. a ticker that gives two permnos: the one whose CRSP share class is the class in the client name
    6. the same permno twice in one list: keep the row whose name matches CRSP, else the CUSIP row
    7. what is left has no permno and is listed in known_gaps
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from pipeline.checks import PASS, WARN, Check, check
from pipeline.config import (
    CLIENT_NO_TICKER,
    NAME_GENERIC_WORDS,
    NAME_STOPWORDS,
    NAMES_FILE,
    DataPaths,
)
from pipeline.outputs import fresh_stage_dir, table_info, write_parquet, write_summary
from pipeline.stages import s00b_client
from pipeline.stages.s00_inventory import read_header

STAGE = "s00c_client_map"
OUTPUT = "client_permno_map.parquet"
CIZ_FILE = "r2000_price_crsp_ciz.csv"
NAMES_COLUMNS = ["permno", "permco", "secinfostartdt", "secinfoenddt", "issuernm", "ticker", "cusip", "sharetype",
                 "securitytype", "shareclass"]
MIN_MATCH_RATE = 0.99


# ---------- Names ----------

def tokens(name) -> list[str]:
    """Upper-case words without punctuation or legal suffixes; runs of single letters are joined (Q R X ->
    QRX)."""
    out, run = [], ""
    for w in re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " ")).split():
        if len(w) == 1:
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return [w for w in out if w not in NAME_STOPWORDS]


def name_ok(a, b) -> bool:
    """Two company names match: one's words contain the other's, the same distinctive first word (or the same
    first two words when the first is generic), one name inside the other (5+ letters), or first words that start
    alike (4+ letters, not generic)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    sa, sb = set(ta), set(tb)
    if sa <= sb or sb <= sa:
        return True
    if ta[0] == tb[0] and (ta[0] not in NAME_GENERIC_WORDS or ta[1:2] == tb[1:2]):
        return True
    ja, jb = "".join(ta), "".join(tb)
    if min(len(ja), len(jb)) >= 5 and (ja in jb or jb in ja):
        return True
    x, y = ta[0], tb[0]
    return min(len(x), len(y)) >= 4 and x not in NAME_GENERIC_WORDS and (x.startswith(y) or y.startswith(x))


def read_names(path) -> pd.DataFrame:
    """CRSP names history: one row per permno and validity range; ids as text."""
    head = {c.strip().lower(): c for c in read_header(path)}
    n = pd.read_csv(path, usecols=[head[c] for c in NAMES_COLUMNS], dtype=str, keep_default_na=False,
                    na_values=[""])
    n.columns = n.columns.str.strip().str.lower()
    for c in ("secinfostartdt", "secinfoenddt"):
        n[c] = pd.to_datetime(n[c])
    n["permno"] = n["permno"].astype("int64")
    n["permco"] = pd.to_numeric(n["permco"], errors="coerce").astype("Int64")
    return n[NAMES_COLUMNS]


class Names:
    """Lookups on the names history the matching rules share."""

    def __init__(self, n: pd.DataFrame):
        self.n = n
        self.all_names = n.dropna(subset=["issuernm"]).groupby("permno")["issuernm"].agg(lambda x: sorted(set(x)))
        self.by_ticker = n.dropna(subset=["ticker"]).groupby("ticker")["permno"].agg(lambda x: sorted(set(x)))
        self._live: dict = {}

    def name_matches(self, name, permno) -> bool:
        """The client name matches any CRSP name the permno ever had."""
        return any(name_ok(name, x) for x in self.all_names.get(int(permno), []))

    def live(self, date) -> set:
        """Permnos with a names row covering the date (trading on it)."""
        if date not in self._live:
            n = self.n
            self._live[date] = set(n.loc[(n["secinfostartdt"] <= date) & (n["secinfoenddt"] >= date), "permno"])
        return self._live[date]

    def valid_on(self, rows: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
        """rows joined to the CRSP rows whose `right` equals rows[left] and are valid on rows.date."""
        crsp = self.n[self.n[right].isin(set(rows[left].dropna()))]
        m = rows.merge(crsp, left_on=left, right_on=right, how="left", suffixes=("", "_crsp"))
        return m[(m["secinfostartdt"] <= m["date"]) & (m["secinfoenddt"] >= m["date"])]

    def suggest(self, name, ticker, date):
        """Rule 4: the single permno with this ticker on any date, trading on the date, whose names match."""
        t = re.sub(r"\.XX\d+$", "", str(ticker))
        cands = [p for p in self.by_ticker.get(t, []) if p in self.live(date) and self.name_matches(name, p)]
        return cands[0] if len(cands) == 1 else None


# ---------- Prices on the last September trading day ----------

def september_prices(path) -> set[tuple[int, int]]:
    """(year, permno) with a CRSP price (a real trading row, DlyPrc not 0) on each year's last September trading
    day. Streams the CIZ file reading 4 columns as text."""
    original = {c.strip().lower(): c for c in read_header(path)}
    cols = [original[c] for c in ("permno", "dlycaldt", "dlydelflg", "dlyprc")]
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=1 << 26),
                            convert_options=pacsv.ConvertOptions(include_columns=cols,
                                                                 column_types={c: pa.string() for c in cols}))
    parts = []
    for batch in reader:
        d = batch.column(original["dlycaldt"])
        keep = pc.and_(pc.equal(pc.utf8_slice_codeunits(d, 5, 7), "09"),
                       pc.greater_equal(pc.utf8_slice_codeunits(d, 8, 10), "21"))
        parts.append(pa.Table.from_batches([batch]).filter(keep).to_pandas())
    px = pd.concat(parts, ignore_index=True)
    px.columns = [c.strip().lower() for c in px.columns]
    px["date"] = pd.to_datetime(px["dlycaldt"])
    last = set(px.groupby(px["date"].dt.year)["date"].max())
    prc = pd.to_numeric(px["dlyprc"], errors="coerce")
    on = px[px["date"].isin(last) & px["dlydelflg"].eq("N") & prc.notna() & prc.ne(0)]
    return set(zip(on["date"].dt.year.astype(int), on["permno"].astype(int)))


# ---------- Matching ----------

def first_match(c: pd.DataFrame, names: Names) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rules 1-2: candidates per row and the status before the review rules."""
    cu = c[c["method"] == "cusip"]
    tk = c[(c["method"] == "ticker") & (c["ticker"] != CLIENT_NO_TICKER)]
    m = pd.concat([names.valid_on(cu, "cusip8", "cusip"), names.valid_on(tk, "ticker", "ticker")],
                  ignore_index=True)
    m["permno"] = m["permno"].astype("int64")
    m["name_ok"] = [names.name_matches(a, p) for a, p in zip(m["name"], m["permno"])]
    cand = m.groupby("row").agg(n_permnos=("permno", "nunique"),
                                candidates=("permno", lambda s: " ".join(map(str, sorted(set(s))))),
                                crsp_names=("issuernm", lambda s: " | ".join(sorted(set(s.dropna())))))
    one = m.drop_duplicates("row").set_index("row")
    c = c.set_index("row").join(cand)
    c["permno"] = one["permno"]
    c["crsp_name"] = one["issuernm"]
    c["name_ok"] = one["name_ok"].astype("object")
    c.loc[c["n_permnos"].ne(1), ["crsp_name", "name_ok"]] = None

    status = pd.Series("matched", index=c.index)
    status[c["n_permnos"].isna()] = "unmatched"
    status[c["ticker"].eq(CLIENT_NO_TICKER) & c["method"].eq("ticker")] = "unmatched"
    status[c["n_permnos"].gt(1)] = "flag_multiple_permnos"
    status[c["method"].eq("ticker") & c["n_permnos"].eq(1) & c["name_ok"].eq(False)] = "flag_name_mismatch"
    c["status"] = status
    c.loc[c["status"] != "matched", "permno"] = None
    return c, m


def unmatched_reasons(c: pd.DataFrame, names: Names) -> pd.DataFrame:
    """Where an unmatched CUSIP or ticker is in CRSP at all, and why it did not match on the 9/30."""
    um = c["status"] == "unmatched"
    for key, col in (("cusip", "cusip8"), ("ticker", "ticker")):
        g = names.n.groupby(key).agg(first=("secinfostartdt", "min"), last=("secinfoenddt", "max"),
                                     permnos=("permno", lambda s: " ".join(map(str, sorted(set(s))))))
        sel = um & c["method"].eq("cusip" if key == "cusip" else "ticker")
        g = g.reindex(c.loc[sel, col])
        c.loc[sel, "crsp_first"] = g["first"].values
        c.loc[sel, "crsp_last"] = g["last"].values
        c.loc[sel, "crsp_permnos_any_date"] = g["permnos"].values

    def reason(r):
        if r.method == "ticker" and r.ticker == CLIENT_NO_TICKER:
            return "no ticker"
        if pd.isna(r.crsp_first):
            return "not in CRSP file"
        if r.date > r.crsp_last:
            return "CRSP validity ended before 9/30"
        if r.date < r.crsp_first:
            return "CRSP validity starts after 9/30"
        return "in CRSP but not valid on 9/30 (gap)"
    c.loc[um, "unmatched_reason"] = c[um].apply(reason, axis=1)
    return c


def review_rules(c: pd.DataFrame, names: Names, priced: set) -> tuple[pd.DataFrame, dict]:
    """Rules 3-7 on the first-match table (indexed by row)."""
    info = {}

    def sug(mask: pd.Series) -> pd.Series:
        rows = c[mask]
        return pd.Series([names.suggest(r.name_, r.ticker, r.date) for r in
                          rows.rename(columns={"name": "name_"}).itertuples()], index=rows.index, dtype="float")
    # rule 4 candidates for ticker rows, rule 3 for CUSIP matches whose name never matched
    fb = c["method"].eq("ticker") & c["status"].ne("matched") & c["ticker"].ne(CLIENT_NO_TICKER)
    c["suggested_permno"] = np.nan
    c.loc[fb, "suggested_permno"] = sug(fb)
    cf = c["method"].eq("cusip") & c["status"].eq("matched") & c["name_ok"].eq(False)
    alt = sug(cf)
    conflict = alt.notna() & alt.ne(c.loc[cf, "permno"].astype("float"))
    rows = conflict[conflict].index
    c.loc[rows, "suggested_permno"] = alt[rows]
    c.loc[rows, "candidates"] = c.loc[rows, "permno"].astype(int).astype(str)
    c.loc[rows, "crsp_names"] = c.loc[rows, "crsp_name"]
    c.loc[rows, "status"] = "flag_cusip_is_other_company"
    c.loc[rows, "permno"] = None

    c["final_permno"] = c["permno"].astype("float")
    c["final_method"] = c["method"].where(c["status"].eq("matched"))
    c["fallback_rejected"] = None
    # rule 3
    a = c["status"].eq("flag_cusip_is_other_company") & c["suggested_permno"].notna()
    c.loc[a, "final_permno"] = c.loc[a, "suggested_permno"]
    c.loc[a, "final_method"] = "cusip_corrected_by_ticker"
    # rule 4, ticker rows, then unmatched CUSIP rows
    has_price = pd.Series([(y, int(p)) in priced if pd.notna(p) else False
                           for y, p in zip(c["year"], c["suggested_permno"])], index=c.index)
    b = c["method"].eq("ticker") & c["status"].isin(["unmatched", "flag_name_mismatch"]) & c["suggested_permno"].notna()
    c.loc[b & has_price, "final_permno"] = c.loc[b & has_price, "suggested_permno"]
    c.loc[b & has_price, "final_method"] = "ticker_any_date"
    c.loc[b & ~has_price, "fallback_rejected"] = "no CRSP price on 9/30"
    b2 = c["method"].eq("cusip") & c["status"].eq("unmatched")
    c.loc[b2, "suggested_permno"] = sug(b2)
    has_price = pd.Series([(y, int(p)) in priced if pd.notna(p) else False
                           for y, p in zip(c["year"], c["suggested_permno"])], index=c.index)
    b2 = b2 & c["suggested_permno"].notna()
    c.loc[b2 & has_price, "final_permno"] = c.loc[b2 & has_price, "suggested_permno"]
    c.loc[b2 & has_price, "final_method"] = "cusip_row_ticker_any_date"
    c.loc[b2 & ~has_price, "fallback_rejected"] = "no CRSP price on 9/30"
    info.update(cusip_corrected=int(a.sum()), ticker_fallback=int(b.sum()), ticker_fallback_priced=int(
        (b & has_price.reindex(c.index)).sum()), cusip_row_fallback=int(b2.sum()))
    # rule 5: the share class named in the client name
    mp = c["status"].eq("flag_multiple_permnos")
    n = names.n
    for i, r in c[mp].iterrows():
        cls = re.search(r"\bCLASS ([A-Z])\b", str(r["name"]).upper())
        if not cls or pd.isna(r["candidates"]):
            continue
        cands = [int(x) for x in r["candidates"].split()]
        v = n[n["permno"].isin(cands) & (n["secinfostartdt"] <= r["date"]) & (n["secinfoenddt"] >= r["date"])
              & n["shareclass"].eq(cls.group(1))]["permno"].unique()
        if len(v) == 1:
            c.loc[i, ["final_permno", "final_method"]] = [float(v[0]), f"ticker_shareclass_{cls.group(1)}"]
    # rule 6
    c["final_name_ok"] = [names.name_matches(nm, p) if pd.notna(p) else None
                          for nm, p in zip(c["name"], c["final_permno"])]
    dup = c["final_permno"].notna() & c.duplicated(["year", "final_permno"], keep=False)
    dropped = 0
    for _, g in c[dup].groupby(["year", "final_permno"]):
        good = g[g["final_name_ok"].eq(True)]
        keep = good.index[:1] if len(good) == 1 else g.sort_values("method", kind="stable").index[:1]
        for i in g.index.difference(keep):
            c.loc[i, ["final_permno", "final_method"]] = [np.nan, "dropped_duplicate"]
            dropped += 1
    info["dropped_duplicates"] = dropped
    # rule 7
    gap = c["final_permno"].isna() & c["final_method"].ne("dropped_duplicate")
    c.loc[gap, "final_method"] = "unmatched"
    c["final_permno"] = c["final_permno"].astype("Int64")
    c["suggested_permno"] = c["suggested_permno"].astype("Int64")
    c["permno"] = c["permno"].astype("Int64")
    return c, info


def known_gaps(c: pd.DataFrame) -> pd.DataFrame:
    """Rule 7: one row per client security without a permno, with the reason."""
    g = c[c["final_method"].eq("unmatched")].copy()
    g["why"] = g["unmatched_reason"].fillna(g["status"]).fillna("")
    rej = g["fallback_rejected"].notna()
    g.loc[rej, "why"] = g.loc[rej, "why"] + "; ticker fallback rejected: no CRSP price on 9/30"
    g.loc[g["status"].eq("flag_name_mismatch") & g["suggested_permno"].isna(), "why"] = \
        "ticker valid on 9/30 belongs to another company; no fallback found"
    sel = g["method"].eq("cusip") & g["suggested_permno"].isna()
    g.loc[sel, "why"] = g.loc[sel, "why"] + "; ticker fallback: no single match"
    g.loc[g["status"].eq("flag_multiple_permnos"), "why"] = "ticker matches 2 permnos; no share class match on 9/30"
    sec = (g.groupby(["name", "ticker_region", "cusip9"], dropna=False)
             .agg(years=("year", lambda s: ", ".join(map(str, sorted(s)))), rows=("year", "size"),
                  why=("why", lambda s: "; ".join(sorted(set(s)))))
             .reset_index().sort_values(["rows", "name"], ascending=[False, True]))
    return sec.reset_index(drop=True)


def gaps_markdown(sec: pd.DataFrame, rows: int) -> str:
    md = ["# Client lists: rows without a permno", "",
          "Written by pipeline stage s00c_client_map. Rows of the client's 9/30 lists that have no permno after "
          "every matching rule (pipeline/PLAN.md, s00c_client_map). One line per security. Licensed data: never "
          "commit this file.", "", f"{rows} rows, {len(sec)} securities.", "",
          "| security | ticker | CUSIP | years | rows | why |", "|---|---|---|---|---|---|"]
    md += [f"| {r.name} | {r.ticker_region} | {r.cusip9} | {r.years} | {r.rows} | {r.why} |" for r in sec.itertuples()]
    return "\n".join(md) + "\n"


def build(c: pd.DataFrame, names: Names, priced: set) -> tuple[pd.DataFrame, dict]:
    c, _ = first_match(c, names)
    c = unmatched_reasons(c, names)
    c, info = review_rules(c, names, priced)
    cols = ["year", "date", "name", "ticker_region", "ticker", "cusip9", "cusip8", "method", "weight", "status",
            "n_permnos", "candidates", "crsp_names", "permno", "crsp_name", "name_ok", "unmatched_reason",
            "crsp_first", "crsp_last", "crsp_permnos_any_date", "suggested_permno", "fallback_rejected",
            "final_permno", "final_method", "final_name_ok"]
    out = c.reset_index()[["row"] + cols]
    for col in ("name_ok", "final_name_ok"):
        out[col] = out[col].astype("boolean")
    return out, info


def by_year(c: pd.DataFrame) -> pd.DataFrame:
    t = c.assign(ok=c["final_permno"].notna(), dropped=c["final_method"].eq("dropped_duplicate"))
    yr = t.groupby("year").agg(rows=("ok", "size"), dropped_duplicates=("dropped", "sum"), with_permno=("ok", "sum"))
    yr["match_rate"] = yr["with_permno"] / (yr["rows"] - yr["dropped_duplicates"])
    return yr


def run_checks(c: pd.DataFrame, lists: pd.DataFrame, yr: pd.DataFrame, priced: set, listed: set | None
               ) -> list[Check]:
    fallback = c["final_method"].isin(["ticker_any_date", "cusip_row_ticker_any_date"])
    unpriced = [(y, p) for y, p in zip(c.loc[fallback, "year"], c.loc[fallback, "final_permno"])
                if (int(y), int(p)) not in priced]
    explained = c["final_permno"].notna() | c["final_method"].isin(["unmatched", "dropped_duplicate"])
    low = yr[yr["match_rate"] < MIN_MATCH_RATE]
    checks = [
        check("one map row per client row", len(c) == len(lists) and set(c["row"]) == set(lists["row"])),
        check("no permno twice in one list", not c.dropna(subset=["final_permno"])
              .duplicated(["year", "final_permno"]).any()),
        check("every row has a permno, or is a known gap or a dropped duplicate", bool(explained.all())),
        check("every ticker fallback has a CRSP price on the last September trading day", not unpriced,
              f"{len(unpriced)} without"),
        check(f"match rate per list at least {MIN_MATCH_RATE:.0%}", low.empty,
              f"{yr['match_rate'].min():.2%} to {yr['match_rate'].max():.2%}" +
              (f"; below: {low['match_rate'].round(4).to_dict()}" if len(low) else ""), level=WARN),
    ]
    if listed is not None:
        missing = set(c["final_permno"].dropna().astype(int)) - listed
        checks.append(check("every mapped permno is in the permno upload list", not missing,
                            f"{len(missing)} not listed (rerun the permno-based WRDS queries)", level=WARN))
    return checks


def run(paths: DataPaths) -> dict:
    out = fresh_stage_dir(paths, STAGE)
    lists = pd.read_parquet(paths.stage_dir(s00b_client.STAGE) / s00b_client.OUTPUT)
    names = Names(read_names(paths.raw / NAMES_FILE))
    priced = september_prices(paths.raw / CIZ_FILE)
    c, info = build(lists, names, priced)
    gaps = known_gaps(c)
    yr = by_year(c)
    listed = (set(int(x) for x in paths.permno_list.read_text().split()) if paths.permno_list.exists() else None)
    checks = run_checks(c, lists, yr, priced, listed)

    write_parquet(c, out / OUTPUT)
    write_parquet(gaps, out / "known_gaps.parquet")
    n_gap_rows = int(c["final_method"].eq("unmatched").sum())
    (out / "known_gaps.md").write_text(gaps_markdown(gaps, n_gap_rows), encoding="utf-8")
    checks.append(Check("rows by final method", PASS, ", ".join(
        f"{k} {v:,}" for k, v in c["final_method"].value_counts().items())))
    extra = {"by_year": {int(y): {k: (round(float(v), 4) if k == "match_rate" else int(v)) for k, v in r.items()}
                         for y, r in yr.iterrows()},
             "final_method": {k: int(v) for k, v in c["final_method"].value_counts().items()},
             "rules": info, "known_gap_rows": n_gap_rows, "known_gap_securities": len(gaps)}
    summary = write_summary(out, STAGE, {OUTPUT: table_info(c, "date"), "known_gaps.parquet": {"rows": len(gaps)}},
                            checks, extra)
    total = yr["with_permno"].sum() / (yr["rows"] - yr["dropped_duplicates"]).sum()
    print(f"  {len(c):,} client rows, {int(c['final_permno'].notna().sum()):,} with a permno ({total:.2%}); "
          f"{n_gap_rows} rows / {len(gaps)} securities without one; {info['dropped_duplicates']} duplicates dropped")
    for ck in checks:
        if ck.status != PASS:
            print(f"  {ck.status}  {ck.name}: {ck.detail}")
    return summary
