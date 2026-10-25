"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/config.py
PURPOSE: Data folder paths (from --data-dir), the raw file specs, and the constants every stage shares.

Data folder layout (see pipeline/README.md):
    raw/                 WRDS and FRED downloads (read only)
    raw/query_inputs/    upload lists used for the WRDS queries (permno_list.txt, ibes_tickers.txt)
    archive/             old notebook and outputs (reference only, never read by the pipeline)
    processed/<stage>/   each stage's outputs (written only by that stage)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------- Paths ----------

@dataclass(frozen=True)
class DataPaths:
    """Folders inside the data folder passed with --data-dir."""
    root: Path

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "DataPaths":
        root = Path(data_dir).expanduser().resolve()
        if not (root / "raw").is_dir():
            raise FileNotFoundError(f"{root} has no raw/ folder; pass the data folder with --data-dir")
        return cls(root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def query_inputs(self) -> Path:
        return self.raw / "query_inputs"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    def stage_dir(self, stage: str) -> Path:
        return self.processed / stage


# ---------- Date ranges of the WRDS pulls (also in the refresh checklist in README.md) ----------

DAILY_START = "2006-12-01"          # CRSP CIZ daily, ETFs, index levels, Fama-French daily
HOLDINGS_START = "2008-02-29"       # first IWM holdings report on CRSP
FUNDAMENTALS_START = "2005-01-01"   # quarterly and annual datadate (TTM and 4-quarter growth need history)
IBES_START = "2005-01-01"           # statpers
FRED_START = "2006-01-01"


# ---------- Raw files ----------

@dataclass(frozen=True)
class RawFile:
    """What s00_inventory checks in one raw file. Column names are lowercase."""
    name: str
    date_col: str | None
    required: tuple[str, ...]
    unique: tuple[str, ...] = ()                    # id columns whose distinct values are counted
    groups: tuple[tuple[str, ...], ...] = ()        # row counts by these columns
    blanks: tuple[str, ...] = ()                    # columns whose missing values ("" or ".") are counted
    expected_values: dict[str, tuple[str, ...]] = field(default_factory=dict)   # all must appear
    first_date_by: str | None = None                # WARN if the first date is later
    upload_list: tuple[str, str] | None = None      # (query_inputs file, id column) the query was run with


_FF_COLS = ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")
IBES_MEASURES = {"EPS": "earnings per share", "CPS": "cash flow per share", "DPS": "dividends per share",
                 "EBT": "EBITDA", "SAL": "sales", "CPX": "capital expenditure", "EBI": "EBIT"}
IBES_FPI = {"0": "long-term growth", "1": "FY1", "2": "FY2", "3": "FY3"}     # FY4 not used (decision 2026-10-25)
FRED_SERIES = {"DGS10": "ust_10y", "DGS3MO": "tbill_3m", "DBAA": "moodys_baa", "DAAA": "moodys_aaa"}
INDEX_TICS = {"I0124": "r2000", "I0120": "r2000_growth", "I0122": "r2000_value"}
ETF_PERMNOS = {88222: "iwm", 88399: "iwn", 88401: "iwo"}

RAW_FILES: tuple[RawFile, ...] = (
    RawFile("iwm_holdings_raw.csv", "report_dt",
            ("crsp_portno", "report_dt", "eff_dt", "percent_tna", "nbr_shares", "security_name", "cusip", "permno",
             "ticker"),
            unique=("permno", "cusip"), groups=(("crsp_portno",),), blanks=("permno", "eff_dt"),
            expected_values={"crsp_portno": ("1004185", "1025818")}, first_date_by=HOLDINGS_START),
    RawFile("r2000_price_crsp_ciz.csv", "dlycaldt",
            ("permno", "hdrcusip", "ticker", "permco", "siccd", "dlycaldt", "dlydelflg", "delreasontype", "dlyprc",
             "dlyret", "dlyvol", "dlyclose", "dlylow", "dlyhigh", "dlyopen", "shrout", "dlycumfacpr", "dlycumfacshr"),
            unique=("permno",), groups=(("dlydelflg",),), blanks=("dlyprc", "dlyret"),
            first_date_by=DAILY_START, upload_list=("permno_list.txt", "permno")),
    RawFile("benchmark_etfs.csv", "dlycaldt", ("permno", "ticker", "dlycaldt", "dlyprc", "dlyret"),
            unique=("permno",), groups=(("ticker",),), blanks=("dlyret",),
            expected_values={"permno": tuple(str(p) for p in ETF_PERMNOS)}, first_date_by=DAILY_START),
    RawFile("delistings.csv", "delistingdt",
            ("permno", "delistingdt", "delactiontype", "delreasontype", "delpaymenttype", "delret"),
            unique=("permno",), groups=(("delactiontype",),), blanks=("delret",),
            upload_list=("permno_list.txt", "permno")),
    RawFile("index_close.csv", "datadate", ("tic", "datadate", "prccd"), groups=(("tic",),), blanks=("prccd",),
            expected_values={"tic": tuple(INDEX_TICS)}, first_date_by=DAILY_START),
    RawFile("crsp_ibes_link.csv", "sdate", ("ticker", "permno", "ncusip", "sdate", "edate", "score"),
            unique=("permno", "ticker"), groups=(("score",),), blanks=("edate",),
            upload_list=("permno_list.txt", "permno")),
    RawFile("ff_daily.csv", "date", ("date",) + _FF_COLS, blanks=_FF_COLS, first_date_by=DAILY_START),
    RawFile("ff_monthly.csv", "dateff", ("dateff",) + _FF_COLS, blanks=_FF_COLS, first_date_by="2006-12-31"),
    RawFile("fundamentals_raw.csv", "datadate",
            ("gvkey", "lpermno", "datadate", "fyearq", "fqtr", "indfmt", "consol", "popsrc", "datafmt", "tic", "conm",
             "ajexq", "curcdq", "datacqtr", "datafqtr", "rdq",
             "actq", "apq", "atq", "ceqq", "cheq", "cogsq", "cshfdq", "cshoq", "dlcq", "dlttq", "dpq", "epsfxq",
             "epspxq", "gdwlq", "ibq", "intanq", "invtq", "lctq", "ltq", "mibtq", "niq", "oiadpq", "oibdpq", "piq",
             "ppentq", "pstkq", "rectq", "revtq", "saleq", "seqq", "txditcq", "txtq", "xintq", "xrdq", "xsgaq",
             "aqcy", "capxy", "dltisy", "dltry", "dvy", "oancfy", "prstkcy", "sstky",
             "costat", "dvpsxq", "prccq", "ggroup", "gind", "gsector", "gsubind", "naics"),
            unique=("lpermno", "gvkey"), groups=(("curcdq",),), blanks=("rdq", "ajexq", "gsector"),
            first_date_by="2005-03-31", upload_list=("permno_list.txt", "lpermno")),
    RawFile("fundamentals_annual_raw.csv", "datadate",
            ("gvkey", "lpermno", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "tic", "conm", "ajex",
             "curcd", "fyr", "act", "ap", "aqc", "at", "capx", "ceq", "che", "cogs", "cshfd", "csho", "dlc", "dltis",
             "dltr", "dltt", "dp", "dv", "dvc", "emp", "epsfx", "epspx", "gdwl", "ib", "intan", "invt", "lct", "lt",
             "mib", "ni", "oancf", "oiadp", "oibdp", "pi", "ppent", "prstkc", "pstk", "rect", "revt", "sale", "seq",
             "sstk", "txditc", "txt", "xint", "xintd", "xrd", "xsga", "costat", "sich", "dvpsx_f", "prcc_f",
             "ggroup", "gind", "gsector", "gsubind", "naics"),
            unique=("lpermno", "gvkey"), groups=(("curcd",),), blanks=("ajex", "oiadp", "dp"),
            first_date_by="2005-12-31", upload_list=("permno_list.txt", "lpermno")),
    RawFile("forward_estimates.csv", "statpers",
            ("ticker", "cusip", "oftic", "statpers", "measure", "fiscalp", "fpi", "curcode", "numest", "numup",
             "numdown", "medest", "meanest", "stdev", "fpedats", "actual", "anndats_act"),
            unique=("ticker",), groups=(("measure", "fpi"), ("curcode",), ("fiscalp",)), blanks=("meanest", "stdev"),
            expected_values={"measure": tuple(IBES_MEASURES), "fpi": tuple(IBES_FPI)}, first_date_by="2005-01-31",
            upload_list=("ibes_tickers.txt", "ticker")),
) + tuple(
    RawFile(f"fred_{series}.csv", "observation_date", ("observation_date", series.lower()),
            blanks=(series.lower(),), first_date_by="2006-01-31")
    for series in FRED_SERIES
)

QUERY_INPUTS = ("permno_list.txt", "ibes_tickers.txt")


# ---------- Cleaning rules (ported from archive/WRDS_Cleaning.ipynb; see PLAN.md) ----------

# Membership: a stock missing from one snapshot stays in its period; missing from 2+ starts a new period
MEMBERSHIP_BRIDGE_SNAPSHOTS = 1

# Holdings lines that are not common stock (ported unchanged from the notebook's membership cell, audit A8)
NON_STOCK_PATTERN = (r"CASH|FUND|FUTURE|FEES|ADJUSTMENT|\bCVR\b|ESCROW|EARNOUT|RIGHTS?\b|WARRANT|GNMA|GOVT|SCH DIS|"
                     r"AUTH|FING|TREASURY|SWAP|MARGIN|COLLATERAL|PAYABLE|RECEIVABLE|CURRENCY|DOLLAR|\bINDEX\b|E-MINI|"
                     r"RUSSELL|STANDARD LIFE|CONTINGENT|LITIGATION|PENDING|\bTBA\b|NOTE|BOND|%|\d{2}-[A-Z]{3}-\d{4}|"
                     r"BANK FINLAND")

# Non-member stocks received in mergers (decision 11): excluded from the universe if (a) the month-end market cap
# is more than NON_MEMBER_CAP_MULTIPLE x the largest other stock in the snapshot, or (b) the stock is in at most
# NON_MEMBER_MAX_SNAPSHOTS snapshots, all with weight 0.00%
NON_MEMBER_CAP_MULTIPLE = 3.0
NON_MEMBER_MAX_SNAPSHOTS = 3

# Membership checks
SNAPSHOT_SIZE_RANGE = (1800, 2150)
SNAPSHOT_WEIGHT_RANGE = (90.0, 110.0)          # percent of the fund
KNOWN_MISSING_MONTHS = ("2009-04", "2009-05", "2009-06")
NEIGHBOUR_MISSING_MAX = 20

# Counts of the notebook's last run (2026-10-22); checks WARN when a stage is more than LAST_RUN_TOLERANCE away
LAST_RUN = {
    "snapshots": 218, "snapshot_rows": 428_354, "permnos": 5_918, "periods": 7_576, "known_gaps": 72,
    "cusip_fill_lines": 1_569, "price_duplicates": 1_801, "daily_rows": 15_309_394, "monthly_rows": 733_698,
    "delisting_fills": 49,
}
LAST_RUN_TOLERANCE = 0.02

# Delisting returns: CIZ dlyret on the flagged row (DlyDelFlg = Y) already includes delret, so it is never added
# again. A missing one is filled by delactiontype: -30% for GDR (dropped for performance), else 0%.
DELISTING_FILL = {"GDR": -0.30}
DELISTING_FILL_DEFAULT = 0.0

# Momentum and volatility windows in trading days; a window needs MIN_OBS_SHARE of its days
MOMENTUM_WINDOWS = {"mom_1m": 21, "mom_6m": 126, "mom_12m": 252}
VOL_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
MIN_OBS_SHARE = 0.8
TRADING_DAYS_PER_YEAR = 252

# Report date when Compustat has no rdq: datadate + 45 days (fiscal Q1-Q3) or + 90 days (Q4 and annual)
REPORT_LAG_DAYS_Q = 45
REPORT_LAG_DAYS_Q4 = 90
REPORT_LAG_DAYS_ANNUAL = 90

# IBES
IBES_LINK_SCORES = (1, 2)
IBES_CURRENCY = "USD"
IBES_MAX_AGE_DAYS = 60          # panel uses the latest statpers within 60 days of the rebalance date
PRICE_ASOF_DAYS = 7             # price on a date = last trading day on or before it, at most 7 days earlier

# Market closed (Hurricane Sandy) but Compustat has index levels; dropped so returns line up with CRSP
CLOSED_MARKET_DAYS = ("2012-10-29", "2012-10-30")

# Sectors from the SIC code as of the date (last non-zero SICCD), checked in this order; anything else is
# Industrials. Major group NN = SIC NN00-NN99.
SECTOR_RULES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Energy", ((1200, 1399), (2900, 2999))),
    ("Utilities", ((4900, 4949),)),
    ("Other", ((9900, 9999),)),                       # also a missing SIC
    ("Technology", ((3500, 3599), (3600, 3699), (3800, 3899), (7300, 7399), (8731, 8734))),
    ("Healthcare", ((2800, 2899), (8000, 8099))),
    ("Financials", ((6000, 6799),)),
    ("Consumer", ((2000, 2399), (5000, 5999), (7000, 7099), (7200, 7299), (7500, 7599), (7800, 7999),
                  (8200, 8299), (8400, 8499))),
)
SECTOR_DEFAULT = "Industrials"
SECTOR_MISSING = "Other"
FOSSIL_SIC_RANGES = ((1200, 1299), (1300, 1399), (2900, 2999), (4920, 4929), (5170, 5179))

UNITS = {
    "returns": "decimals (0.01 = 1%)",
    "yields": "decimals per year (FRED percent / 100)",
    "mkt_cap_musd": "$ millions",
    "mkt_cap (daily prices only)": "$ thousands (prc x shrout, CRSP shrout in thousands)",
    "Compustat amounts": "$ millions",
    "Compustat shares (cshoq, cshfdq)": "millions",
    "prices and per-share amounts": "$ per share",
    "index_weight": "percent of the IWM fund",
    "volatility": "annualized decimal",
}

PARQUET_COMPRESSION = "zstd"
