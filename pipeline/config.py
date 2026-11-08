"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/config.py
PURPOSE: Data folder paths (from --data-dir), the raw file specs, and the constants every stage shares.

Data folder layout (see pipeline/README.md):
    raw/                 WRDS and FRED downloads (read only)
    raw/query_inputs/    upload lists used for the WRDS queries (permno_list.txt, ibes_tickers.txt)
    raw/client/          the client's Russell 2000 lists (licensed, never committed)
    archive/             old notebook and outputs (reference only, never read by the pipeline)
    processed/<stage>/   each stage's outputs (written only by that stage)

Every folder and upload list can be moved with run.py options (--raw-dir, --processed-dir, --client-dir,
--permno-list, --ibes-tickers), so a second data version (raw_v2/ -> processed_v2/) runs in place.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

# ---------- Paths ----------

PERMNO_LIST = "permno_list.txt"
IBES_TICKERS = "ibes_tickers.txt"


def _resolve(root: Path, value: str | Path | None) -> Path | None:
    """A path option: absolute, or relative to the data folder."""
    if value is None:
        return None
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


@dataclass(frozen=True)
class DataPaths:
    """Folders inside the data folder passed with --data-dir. Every other location defaults to the data folder's
    standard layout; relative overrides are relative to the data folder."""
    root: Path
    processed_dir: Path | None = None   # default <root>/processed; s09 reruns everything into a temporary one
    raw_dir: Path | None = None         # default <root>/raw
    client_dir: Path | None = None      # default <raw>/client
    permno_list_file: Path | None = None    # default <raw>/query_inputs/permno_list.txt
    ibes_tickers_file: Path | None = None   # default <raw>/query_inputs/ibes_tickers.txt

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, processed_dir: str | Path | None = None, *,
                      raw_dir: str | Path | None = None, client_dir: str | Path | None = None,
                      permno_list: str | Path | None = None,
                      ibes_tickers: str | Path | None = None) -> "DataPaths":
        root = Path(data_dir).expanduser().resolve()
        paths = cls(root, _resolve(root, processed_dir), _resolve(root, raw_dir), _resolve(root, client_dir),
                    _resolve(root, permno_list), _resolve(root, ibes_tickers))
        if not paths.raw.is_dir():
            raise FileNotFoundError(f"{paths.raw} is not a folder; pass the data folder with --data-dir "
                                    "(and the raw folder with --raw-dir if it is not <data-dir>/raw)")
        return paths

    @property
    def raw(self) -> Path:
        return self.raw_dir or self.root / "raw"

    @property
    def query_inputs(self) -> Path:
        return self.raw / "query_inputs"

    @property
    def client(self) -> Path:
        return self.client_dir or self.raw / "client"

    @property
    def permno_list(self) -> Path:
        return self.permno_list_file or self.query_inputs / PERMNO_LIST

    @property
    def ibes_tickers(self) -> Path:
        return self.ibes_tickers_file or self.query_inputs / IBES_TICKERS

    def list_path(self, name: str) -> Path:
        """The upload list behind a QUERY_INPUTS name."""
        return {PERMNO_LIST: self.permno_list, IBES_TICKERS: self.ibes_tickers}[name]

    @property
    def processed(self) -> Path:
        return self.processed_dir or self.root / "processed"

    def stage_dir(self, stage: str) -> Path:
        return self.processed / STAGE_FOLDERS.get(stage, stage)

    def relative(self, path: Path) -> str:
        """`path` relative to the data folder when it is inside it (posix), else absolute."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path.resolve())

    def cli_args(self) -> list[str]:
        """The run.py options that rebuild these paths (s09 reruns every stage with them)."""
        args = ["--data-dir", str(self.root)]
        for opt, value in (("--raw-dir", self.raw_dir), ("--client-dir", self.client_dir),
                           ("--permno-list", self.permno_list_file), ("--ibes-tickers", self.ibes_tickers_file)):
            if value is not None:
                args += [opt, str(value)]
        return args


STAGE_FOLDERS = {"s09_validate": "s09_validation"}       # stages whose folder is not named after them
OLD_PANEL_PATHS = ("archive/r2000_monthly_panel.parquet",  # the notebook's panel, for s09's old vs new backtest
                   "scratch/notebook_ref/r2000_monthly_panel.parquet")


# ---------- Date ranges of the WRDS pulls (also in the refresh checklist in README.md) ----------

# Data version 2 (decision 35): membership from 2002-07, so prices start 1997-09 (60-month beta from 2002) and the
# market series 1997-01. s00 WARNs when a raw file starts after its target.
PRICES_START = "1997-09-01"         # CRSP CIZ daily stock file
ETF_START = "2000-05-01"            # IWM listed 2000-05; IWN and IWO start 2000-07
MARKET_START = "1997-01-01"         # index levels, Fama-French daily and monthly, FRED; market_daily starts here
HOLDINGS_START = "2008-02-29"       # first IWM holdings report on CRSP
FUNDAMENTALS_START = "1999-01-01"   # quarterly and annual datadate (TTM and 4-quarter growth need history)
IBES_START = "2001-01-01"           # statpers
FRED_START = MARKET_START
CLIENT_LISTS_START = "2002-09-30"   # first client list (9/30/2002)


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
    optional: tuple[str, ...] = ()                  # columns kept when present (WARN when missing)
    folder: str = ""                                # "client": in the client folder instead of raw/
    skip_lines: int = 0                             # title lines above the header row
    date_format: str | None = None                  # strptime format when the date column is not YYYY-MM-DD


_FF_COLS = ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")
IBES_MEASURES = {"EPS": "earnings per share", "CPS": "cash flow per share", "DPS": "dividends per share",
                 "EBT": "EBITDA", "SAL": "sales", "CPX": "capital expenditure", "EBI": "EBIT"}
IBES_FPI = {"0": "long-term growth", "1": "FY1", "2": "FY2", "3": "FY3"}     # FY4 not used (decision 2026-10-25)
FRED_SERIES = {"DGS10": "ust_10y", "DGS3MO": "tbill_3m", "DBAA": "moodys_baa", "DAAA": "moodys_aaa"}
INDEX_TICS = {"I0124": "r2000", "I0120": "r2000_growth", "I0122": "r2000_value"}
ETF_PERMNOS = {88222: "iwm", 88399: "iwn", 88401: "iwo"}
ETF_FIRST_BY = {"iwm": "2000-05-31", "iwn": "2000-07-31", "iwo": "2000-07-31"}     # IWM listed 2000-05, IWN/IWO 2000-07

# Every column a stage reads is required (s00 FAILs when one is missing). Other columns in the files are allowed
# and ignored (decision 37); `optional` ones are kept when present.
CLIENT_FILE = "FR2000 Annual Quant Data(Data).csv"
CLIENT_COLUMNS = {"security name": "name", "ticker-region": "ticker_region", "cusip": "cusip9", "date": "date",
                  "russell 2000 port. weight": "weight"}      # client file column (lowercase) -> s00b column
NAMES_FILE = "crsp_names_history.csv"

RAW_FILES: tuple[RawFile, ...] = (
    RawFile("iwm_holdings_raw.csv", "report_dt",
            ("crsp_portno", "report_dt", "eff_dt", "percent_tna", "nbr_shares", "security_name", "cusip", "permno",
             "ticker"),
            unique=("permno", "cusip"), groups=(("crsp_portno",),), blanks=("permno", "eff_dt"),
            expected_values={"crsp_portno": ("1004185", "1025818")}, first_date_by=HOLDINGS_START),
    RawFile("r2000_price_crsp_ciz.csv", "dlycaldt",
            ("permno", "hdrcusip", "ticker", "permco", "siccd", "dlycaldt", "dlydelflg", "delreasontype", "securitynm",
             "dlyprc", "dlyret", "dlyvol", "dlyclose", "dlylow", "dlyhigh", "dlyopen", "shrout", "dlycumfacpr",
             "dlycumfacshr"),
            unique=("permno",), groups=(("dlydelflg",),), blanks=("dlyprc", "dlyret"),
            first_date_by=PRICES_START, upload_list=(PERMNO_LIST, "permno"), optional=("primaryexch",)),
    RawFile("benchmark_etfs.csv", "dlycaldt", ("permno", "ticker", "dlycaldt", "dlyprc", "dlyret"),
            unique=("permno",), groups=(("ticker",),), blanks=("dlyret",),
            expected_values={"permno": tuple(str(p) for p in ETF_PERMNOS)}, first_date_by="2000-05-31"),
    RawFile("delistings.csv", "delistingdt",
            ("permno", "delistingdt", "delactiontype", "delreasontype", "delpaymenttype", "delret"),
            unique=("permno",), groups=(("delactiontype",),), blanks=("delret",),
            upload_list=(PERMNO_LIST, "permno")),
    RawFile("index_close.csv", "datadate", ("tic", "datadate", "prccd"), groups=(("tic",),), blanks=("prccd",),
            expected_values={"tic": tuple(INDEX_TICS)}, first_date_by="1997-01-31"),
    RawFile("crsp_ibes_link.csv", "sdate", ("ticker", "permno", "ncusip", "sdate", "edate", "score"),
            unique=("permno", "ticker"), groups=(("score",),), blanks=("edate",),
            upload_list=(PERMNO_LIST, "permno")),
    RawFile("ff_daily.csv", "date", ("date",) + _FF_COLS, blanks=_FF_COLS, first_date_by="1997-01-31"),
    RawFile("ff_monthly.csv", "dateff", ("dateff",) + _FF_COLS, blanks=_FF_COLS, first_date_by="1997-01-31"),
    RawFile("fundamentals_raw.csv", "datadate",
            ("gvkey", "lpermno", "datadate", "fyearq", "fqtr", "indfmt", "consol", "popsrc", "datafmt", "tic", "conm",
             "ajexq", "curcdq", "datacqtr", "datafqtr", "rdq",
             "actq", "apq", "atq", "ceqq", "cheq", "cogsq", "cshfdq", "cshoq", "dlcq", "dlttq", "dpq", "epsfxq",
             "epspxq", "gdwlq", "ibq", "intanq", "invtq", "lctq", "ltq", "mibtq", "niq", "oiadpq", "oibdpq", "piq",
             "ppentq", "pstkq", "rectq", "revtq", "saleq", "seqq", "txditcq", "txtq", "xintq", "xrdq", "xsgaq",
             "aqcy", "capxy", "dltisy", "dltry", "dvy", "oancfy", "prstkcy", "sstky",
             "costat", "dvpsxq", "prccq", "ggroup", "gind", "gsector", "gsubind", "naics"),
            unique=("lpermno", "gvkey"), groups=(("curcdq",),), blanks=("rdq", "ajexq", "gsector"),
            first_date_by="1999-03-31", upload_list=(PERMNO_LIST, "lpermno")),
    RawFile("fundamentals_annual_raw.csv", "datadate",
            ("gvkey", "lpermno", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt", "tic", "conm", "ajex",
             "curcd", "fyr", "act", "ap", "aqc", "at", "capx", "ceq", "che", "cogs", "cshfd", "csho", "dlc", "dltis",
             "dltr", "dltt", "dp", "dv", "dvc", "emp", "epsfx", "epspx", "gdwl", "ib", "intan", "invt", "lct", "lt",
             "mib", "ni", "oancf", "oiadp", "oibdp", "pi", "ppent", "prstkc", "pstk", "rect", "revt", "sale", "seq",
             "sstk", "txditc", "txt", "xint", "xintd", "xrd", "xsga", "costat", "sich", "dvpsx_f", "prcc_f",
             "ggroup", "gind", "gsector", "gsubind", "naics"),
            unique=("lpermno", "gvkey"), groups=(("curcd",),), blanks=("ajex", "oiadp", "dp"),
            first_date_by="1999-12-31", upload_list=(PERMNO_LIST, "lpermno")),
    RawFile("forward_estimates.csv", "statpers",
            ("ticker", "cusip", "oftic", "statpers", "measure", "fiscalp", "fpi", "curcode", "numest", "numup",
             "numdown", "medest", "meanest", "stdev", "highest", "lowest", "fpedats", "actual", "anndats_act"),
            unique=("ticker",), groups=(("measure", "fpi"), ("curcode",), ("fiscalp",)), blanks=("meanest", "stdev"),
            expected_values={"measure": tuple(IBES_MEASURES), "fpi": tuple(IBES_FPI)}, first_date_by="2001-01-31",
            upload_list=(IBES_TICKERS, "ticker")),
    RawFile(NAMES_FILE, "secinfostartdt",
            ("permno", "permco", "secinfostartdt", "secinfoenddt", "cusip", "ticker", "issuernm", "shareclass",
             "securitytype", "sharetype"),
            unique=("permno", "cusip")),
    RawFile(CLIENT_FILE, "date", tuple(CLIENT_COLUMNS), unique=("cusip",), groups=(("date",),),
            blanks=("ticker-region", "cusip", "russell 2000 port. weight"), first_date_by=CLIENT_LISTS_START,
            folder="client", skip_lines=2, date_format="%m/%d/%Y"),
) + tuple(
    RawFile(f"fred_{series}.csv", "observation_date", ("observation_date", series.lower()),
            blanks=(series.lower(),), first_date_by="1997-01-31")
    for series in FRED_SERIES
)

QUERY_INPUTS = (PERMNO_LIST, IBES_TICKERS)


# ---------- Cleaning rules (ported from archive/WRDS_Cleaning.ipynb; see PLAN.md) ----------

# Membership: a stock missing from one snapshot stays in its period; missing from 2+ starts a new period
MEMBERSHIP_BRIDGE_SNAPSHOTS = 1

# Holdings lines that are not common stock (ported unchanged from the notebook's membership cell, audit A8)
NON_STOCK_PATTERN = (r"CASH|FUND|FUTURE|FEES|ADJUSTMENT|\bCVR\b|ESCROW|EARNOUT|RIGHTS?\b|WARRANT|GNMA|GOVT|SCH DIS|"
                     r"AUTH|FING|TREASURY|SWAP|MARGIN|COLLATERAL|PAYABLE|RECEIVABLE|CURRENCY|DOLLAR|\bINDEX\b|E-MINI|"
                     r"RUSSELL|STANDARD LIFE|CONTINGENT|LITIGATION|PENDING|\bTBA\b|NOTE|BOND|%|\d{2}-[A-Z]{3}-\d{4}|"
                     r"BANK FINLAND")

# Non-member stocks received in mergers (decision 11, revised): a stock-snapshot is excluded from the universe if
# (a) its month-end market cap is more than NON_MEMBER_CAP_MULTIPLE x the largest other stock in the snapshot and
#     its weight is below NON_MEMBER_MAX_WEIGHT_A percent, or
# (b) the stock is in at most NON_MEMBER_MAX_SNAPSHOTS snapshots, all with weight 0.00%, and its market cap is
#     above the snapshot's NON_MEMBER_CAP_QUANTILE_B quantile
NON_MEMBER_CAP_MULTIPLE = 3.0
NON_MEMBER_MAX_WEIGHT_A = 0.05
NON_MEMBER_MAX_SNAPSHOTS = 3
NON_MEMBER_CAP_QUANTILE_B = 0.95

# Membership checks
SNAPSHOT_SIZE_RANGE = (1800, 2150)
SNAPSHOT_WEIGHT_RANGE = (90.0, 110.0)          # percent of the fund
KNOWN_MISSING_MONTHS = ("2009-04", "2009-05", "2009-06")
NEIGHBOUR_MISSING_MAX = 20

# Membership sources (decision 33). membership_source on snapshots and universe periods:
#   client_list  before IWM_FIRST_MONTH: the client's 9/30 lists only
#   iwm          IWM holdings
#   client_fill  IWM_FIRST_MONTH..CLIENT_FILL_LAST_MONTH: client-list members missing from the IWM snapshot (the old
#                fund id 1004185 hides some members behind relabelled security records)
MEMBERSHIP_SOURCES = ("client_list", "iwm", "client_fill")
CLIENT_LIST_FIRST_MONTH = "2002-07"     # the 9/30/2002 list counts from the 2002-07 month end
IWM_FIRST_MONTH = "2008-02"
CLIENT_FILL_LAST_MONTH = "2010-05"


def reconstitution_date(year: int) -> _dt.date:
    """Russell's annual reconstitution: the last Friday of June, or the Friday before when that is the 29th or
    30th. A stock first trading after it (an IPO added after June) counts only from its list's 9/30."""
    day = _dt.date(year, 6, 30)
    day -= _dt.timedelta(days=(day.weekday() - 4) % 7)
    return day - _dt.timedelta(days=7) if day.day >= 29 else day


def client_list_year(month_end) -> int:
    """The client list (its 9/30 year) in force at a month end (the month's last CRSP trading day): each list
    counts from the first month end after that year's reconstitution, so June in most years and July when the
    reconstitution is on June's last trading day (e.g. 2002)."""
    day = month_end.date() if isinstance(month_end, _dt.datetime) else month_end
    return day.year if day > reconstitution_date(day.year) else day.year - 1


# Client list mapping to permnos (s00c_client_map, decision 34)
CLIENT_LIST_MONTH_DAY = (9, 30)
CLIENT_NO_CUSIP = "#Calc"               # FactSet's placeholder when it has no CUSIP: matched by ticker + name
CLIENT_NO_TICKER = "--"
NAME_STOPWORDS = frozenset({
    "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "HOLDINGS", "HOLDING", "HLDGS", "GROUP",
    "GRP", "THE", "LP", "LLC", "NV", "SA", "AG", "CL", "CLASS", "A", "B", "C", "NEW", "DEL", "OF", "COM",
    "INCORPORATED", "NON", "VOTING", "SHARES", "SHS", "ORD", "AND"})
NAME_GENERIC_WORDS = frozenset({
    "FIRST", "AMERICAN", "AMERICA", "NATIONAL", "UNITED", "CENTRAL", "GENERAL", "INTERNATIONAL", "US", "USA",
    "NORTH", "SOUTH", "WEST", "EAST", "PACIFIC", "ATLANTIC", "CITIZENS", "COMMUNITY", "SECURITY", "CAPITAL", "BANK",
    "HOME", "PEOPLES", "GREAT", "SOUTHERN", "NORTHERN", "WESTERN", "EASTERN", "MIDWEST", "TRUST"})

# IWM known-gap fixes (s01 rule 2b): a gap CUSIP gets the permno CRSP gives it while the permno traded during the
# gap's IWM dates, else a client-list match whose 9/30 is within this many months of those dates; a gap whose
# every line holds fewer than GAP_FIX_MIN_SHARES shares is a stray line and is not fixed
GAP_FIX_CLIENT_WINDOW_MONTHS = 3
GAP_FIX_MIN_SHARES = 100

# Counts and factor coverage of the notebook's last run, which the stages compare with. They describe the licensed
# data, so they live next to it (<data-dir>/pipeline_last_run.json: {"counts": {...}, "coverage_2015": {...}}), not in
# this public repo; run.py loads them. Without the file the comparisons report "no last-run value".
LAST_RUN_FILE = "pipeline_last_run.json"
LAST_RUN: dict[str, int] = {}
LAST_RUN_COVERAGE_2015: dict[str, float] = {}
LAST_RUN_TOLERANCE = 0.02


def load_last_run(path: Path) -> bool:
    """Fill LAST_RUN and LAST_RUN_COVERAGE_2015 in place from the data folder's file; False when it is missing."""
    import json
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    LAST_RUN.clear()
    LAST_RUN.update(data.get("counts", {}))
    LAST_RUN_COVERAGE_2015.clear()
    LAST_RUN_COVERAGE_2015.update(data.get("coverage_2015", {}))
    return True


COVERAGE_TOLERANCE = 0.05

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

# Fundamentals (s03, s04)
TAX_RATE_RANGE = (0.0, 0.60)        # effective tax rate = taxes / pretax income, only when pretax income > 0
PRICE_FLOOR = 0.01                  # below this price, price-based ratios are missing (decision 14)
# The quarterly factors whose denominator or numerator is the share price or market cap (decisions 2 and 14)
PRICE_BASED_FACTORS = ("f_book_to_price", "f_earnings_to_price", "f_sales_to_price", "f_ebitda_to_ev",
                       "f_dividend_yield")
# Year-to-date cash flow items converted to quarterly values: raw column -> quarterly column
YTD_ITEMS = {"oancfy": "oancf_q", "capxy": "capx_q", "dvy": "dv_q", "sstky": "sstk_q", "prstkcy": "prstkc_q",
             "dltisy": "dltis_q", "dltry": "dltr_q", "aqcy": "aqc_q"}
SUE_HISTORY_QUARTERS = 8            # SUE is scaled by the std of the previous 8 four-quarter EPS changes
SUE_MIN_HISTORY = 6                 # ... and needs at least 6 of them
SUE_MAX_ABS = 50                    # |SUE| above this is NaN (counted)

# GICS names, all four levels (decision 39). Compustat's GICS is each company's latest classification (current for
# active companies, the last one used for inactive ones, so discontinued codes appear). Names follow the 2023
# structure; discontinued codes keep their last name, marked with the year they were dropped or renumbered.
GICS_SECTORS = {10: "Energy", 15: "Materials", 20: "Industrials", 25: "Consumer Discretionary",
                30: "Consumer Staples", 35: "Health Care", 40: "Financials", 45: "Information Technology",
                50: "Communication Services", 55: "Utilities", 60: "Real Estate"}
GICS_INDUSTRIES = {
    101010: "Energy Equipment & Services", 101020: "Oil, Gas & Consumable Fuels",
    151010: "Chemicals", 151020: "Construction Materials", 151030: "Containers & Packaging",
    151040: "Metals & Mining", 151050: "Paper & Forest Products",
    201010: "Aerospace & Defense", 201020: "Building Products", 201030: "Construction & Engineering",
    201040: "Electrical Equipment", 201050: "Industrial Conglomerates", 201060: "Machinery",
    201070: "Trading Companies & Distributors", 202010: "Commercial Services & Supplies",
    202020: "Professional Services", 203010: "Air Freight & Logistics", 203020: "Passenger Airlines",
    203030: "Marine Transportation", 203040: "Ground Transportation", 203050: "Transportation Infrastructure",
    251010: "Automobile Components", 251020: "Automobiles", 252010: "Household Durables",
    252020: "Leisure Products", 252030: "Textiles, Apparel & Luxury Goods", 253010: "Hotels, Restaurants & Leisure",
    253020: "Diversified Consumer Services", 254010: "Media (discontinued 2018)", 255010: "Distributors",
    255020: "Internet & Direct Marketing Retail (discontinued 2023)", 255030: "Broadline Retail",
    255040: "Specialty Retail",
    301010: "Consumer Staples Distribution & Retail", 302010: "Beverages", 302020: "Food Products",
    302030: "Tobacco", 303010: "Household Products", 303020: "Personal Care Products",
    351010: "Health Care Equipment & Supplies", 351020: "Health Care Providers & Services",
    351030: "Health Care Technology", 352010: "Biotechnology", 352020: "Pharmaceuticals",
    352030: "Life Sciences Tools & Services",
    401010: "Banks", 401020: "Thrifts & Mortgage Finance (discontinued 2023)", 402010: "Financial Services",
    402020: "Consumer Finance", 402030: "Capital Markets", 402040: "Mortgage Real Estate Investment Trusts (REITs)",
    403010: "Insurance", 404020: "Real Estate Investment Trusts (discontinued 2016)",
    404030: "Real Estate Management & Development (discontinued 2016)",
    451010: "Internet Software & Services (discontinued 2018)", 451020: "IT Services", 451030: "Software",
    452010: "Communications Equipment", 452020: "Technology Hardware, Storage & Peripherals",
    452030: "Electronic Equipment, Instruments & Components", 453010: "Semiconductors & Semiconductor Equipment",
    501010: "Diversified Telecommunication Services", 501020: "Wireless Telecommunication Services",
    502010: "Media", 502020: "Entertainment", 502030: "Interactive Media & Services",
    551010: "Electric Utilities", 551020: "Gas Utilities", 551030: "Multi-Utilities", 551040: "Water Utilities",
    551050: "Independent Power and Renewable Electricity Producers",
    601010: "Diversified REITs", 601025: "Industrial REITs", 601030: "Hotel & Resort REITs",
    601040: "Office REITs", 601050: "Health Care REITs", 601060: "Residential REITs", 601070: "Retail REITs",
    601080: "Specialized REITs", 602010: "Real Estate Management & Development",
    601020: "Real Estate Management & Development (2016-2023 code, renumbered 602010)",
    404010: "Real Estate (discontinued 2006)", 452050: "Semiconductor Equipment & Products (discontinued 2002)",
}
GICS_GROUPS = {
    1010: "Energy", 1510: "Materials", 2010: "Capital Goods", 2020: "Commercial & Professional Services",
    2030: "Transportation", 2510: "Automobiles & Components", 2520: "Consumer Durables & Apparel",
    2530: "Consumer Services", 2540: "Media (discontinued 2018)",
    2550: "Consumer Discretionary Distribution & Retail", 3010: "Consumer Staples Distribution & Retail",
    3020: "Food, Beverage & Tobacco", 3030: "Household & Personal Products", 3510: "Health Care Equipment & Services",
    3520: "Pharmaceuticals, Biotechnology & Life Sciences", 4010: "Banks", 4020: "Financial Services",
    4030: "Insurance", 4040: "Real Estate (discontinued 2016)", 4510: "Software & Services",
    4520: "Technology Hardware & Equipment", 4530: "Semiconductors & Semiconductor Equipment",
    5010: "Telecommunication Services", 5020: "Media & Entertainment", 5510: "Utilities",
    6010: "Equity Real Estate Investment Trusts (REITs)", 6020: "Real Estate Management & Development",
}
GICS_SUB_INDUSTRIES = {
    10101010: "Oil & Gas Drilling", 10101020: "Oil & Gas Equipment & Services", 10102010: "Integrated Oil & Gas",
    10102020: "Oil & Gas Exploration & Production", 10102030: "Oil & Gas Refining & Marketing",
    10102040: "Oil & Gas Storage & Transportation", 10102050: "Coal & Consumable Fuels",
    15101010: "Commodity Chemicals", 15101020: "Diversified Chemicals",
    15101030: "Fertilizers & Agricultural Chemicals",
    15101040: "Industrial Gases", 15101050: "Specialty Chemicals", 15102010: "Construction Materials",
    15103010: "Metal, Glass & Plastic Containers", 15103020: "Paper & Plastic Packaging Products & Materials",
    15104010: "Aluminum", 15104020: "Diversified Metals & Mining", 15104025: "Copper", 15104030: "Gold",
    15104040: "Precious Metals & Minerals", 15104045: "Silver", 15104050: "Steel", 15105010: "Forest Products",
    15105020: "Paper Products",
    20101010: "Aerospace & Defense", 20102010: "Building Products", 20103010: "Construction & Engineering",
    20104010: "Electrical Components & Equipment", 20104020: "Heavy Electrical Equipment",
    20105010: "Industrial Conglomerates", 20106010: "Construction Machinery & Heavy Transportation Equipment",
    20106015: "Agricultural & Farm Machinery", 20106020: "Industrial Machinery & Supplies & Components",
    20107010: "Trading Companies & Distributors", 20201010: "Commercial Printing",
    20201030: "Diversified Commercial & Professional Services (discontinued 2008)",
    20201040: "Human Resource & Employment Services (discontinued 2008)",
    20201050: "Environmental & Facilities Services", 20201060: "Office Services & Supplies",
    20201070: "Diversified Support Services", 20201080: "Security & Alarm Services",
    20202010: "Human Resource & Employment Services", 20202020: "Research & Consulting Services",
    20202030: "Data Processing & Outsourced Services", 20301010: "Air Freight & Logistics",
    20302010: "Passenger Airlines", 20303010: "Marine Transportation", 20304010: "Rail Transportation",
    20304020: "Trucking (discontinued 2023)", 20304030: "Cargo Ground Transportation",
    20304040: "Passenger Ground Transportation", 20305010: "Airport Services", 20305020: "Highways & Railtracks",
    20305030: "Marine Ports & Services",
    25101010: "Automotive Parts & Equipment", 25101020: "Tires & Rubber", 25102010: "Automobile Manufacturers",
    25102020: "Motorcycle Manufacturers", 25201010: "Consumer Electronics", 25201020: "Home Furnishings",
    25201030: "Homebuilding", 25201040: "Household Appliances", 25201050: "Housewares & Specialties",
    25202010: "Leisure Products", 25202020: "Photographic Products (discontinued 2014)",
    25203010: "Apparel, Accessories & Luxury Goods", 25203020: "Footwear", 25203030: "Textiles",
    25301010: "Casinos & Gaming", 25301020: "Hotels, Resorts & Cruise Lines", 25301030: "Leisure Facilities",
    25301040: "Restaurants", 25302010: "Education Services", 25302020: "Specialized Consumer Services",
    25401010: "Advertising (discontinued 2018)", 25401020: "Broadcasting (discontinued 2018)",
    25401025: "Cable & Satellite (discontinued 2018)", 25401030: "Movies & Entertainment (discontinued 2018)",
    25401040: "Publishing (discontinued 2018)", 25501010: "Distributors",
    25502010: "Catalog Retail (discontinued 2018)", 25502020: "Internet & Direct Marketing Retail (discontinued 2023)",
    25503010: "Department Stores (discontinued 2023)", 25503020: "General Merchandise Stores (discontinued 2023)",
    25503030: "Broadline Retail", 25504010: "Apparel Retail", 25504020: "Computer & Electronics Retail",
    25504030: "Home Improvement Retail", 25504040: "Other Specialty Retail", 25504050: "Automotive Retail",
    25504060: "Homefurnishing Retail",
    30101010: "Drug Retail", 30101020: "Food Distributors", 30101030: "Food Retail",
    30101040: "Consumer Staples Merchandise Retail", 30201010: "Brewers", 30201020: "Distillers & Vintners",
    30201030: "Soft Drinks & Non-alcoholic Beverages", 30202010: "Agricultural Products & Services",
    30202030: "Packaged Foods & Meats", 30203010: "Tobacco", 30301010: "Household Products",
    30302010: "Personal Care Products",
    35101010: "Health Care Equipment", 35101020: "Health Care Supplies", 35102010: "Health Care Distributors",
    35102015: "Health Care Services", 35102020: "Health Care Facilities", 35102030: "Managed Health Care",
    35103010: "Health Care Technology", 35201010: "Biotechnology", 35202010: "Pharmaceuticals",
    35203010: "Life Sciences Tools & Services",
    40101010: "Diversified Banks", 40101015: "Regional Banks",
    40102010: "Thrifts & Mortgage Finance (discontinued 2023)", 40201020: "Diversified Financial Services",
    40201030: "Multi-Sector Holdings", 40201040: "Specialized Finance",
    40201050: "Commercial & Residential Mortgage Finance", 40201060: "Transaction & Payment Processing Services",
    40202010: "Consumer Finance", 40203010: "Asset Management & Custody Banks",
    40203020: "Investment Banking & Brokerage", 40203030: "Diversified Capital Markets",
    40203040: "Financial Exchanges & Data", 40204010: "Mortgage REITs", 40301010: "Insurance Brokers",
    40301020: "Life & Health Insurance", 40301030: "Multi-line Insurance", 40301040: "Property & Casualty Insurance",
    40301050: "Reinsurance",
    40401010: "Real Estate Investment Trusts (discontinued 2006)",
    40401020: "Real Estate Management & Development (discontinued 2006)",
    40402010: "Diversified REITs (discontinued 2016)", 40402020: "Industrial REITs (discontinued 2016)",
    40402030: "Mortgage REITs (discontinued 2016)", 40402035: "Hotel & Resort REITs (discontinued 2016)",
    40402040: "Office REITs (discontinued 2016)", 40402045: "Health Care REITs (discontinued 2016)",
    40402050: "Residential REITs (discontinued 2016)", 40402060: "Retail REITs (discontinued 2016)",
    40402070: "Specialized REITs (discontinued 2016)",
    40403010: "Diversified Real Estate Activities (discontinued 2016)",
    40403020: "Real Estate Operating Companies (discontinued 2016)",
    40403030: "Real Estate Development (discontinued 2016)", 40403040: "Real Estate Services (discontinued 2016)",
    45101010: "Internet Software & Services (discontinued 2018)", 45102010: "IT Consulting & Other Services",
    45102020: "Data Processing & Outsourced Services (discontinued 2023)",
    45102030: "Internet Services & Infrastructure", 45103010: "Application Software", 45103020: "Systems Software",
    45103030: "Home Entertainment Software (discontinued 2018)", 45201020: "Communications Equipment",
    45202010: "Computer Hardware (discontinued 2014)", 45202020: "Computer Storage & Peripherals (discontinued 2014)",
    45202030: "Technology Hardware, Storage & Peripherals", 45203010: "Electronic Equipment & Instruments",
    45203015: "Electronic Components", 45203020: "Electronic Manufacturing Services",
    45203030: "Technology Distributors", 45205010: "Semiconductor Equipment (discontinued 2002)",
    45205020: "Semiconductors (discontinued 2002)", 45301010: "Semiconductor Materials & Equipment",
    45301020: "Semiconductors",
    50101010: "Alternative Carriers", 50101020: "Integrated Telecommunication Services",
    50102010: "Wireless Telecommunication Services", 50201010: "Advertising", 50201020: "Broadcasting",
    50201030: "Cable & Satellite", 50201040: "Publishing", 50202010: "Movies & Entertainment",
    50202020: "Interactive Home Entertainment", 50203010: "Interactive Media & Services",
    55101010: "Electric Utilities", 55102010: "Gas Utilities", 55103010: "Multi-Utilities", 55104010: "Water Utilities",
    55105010: "Independent Power Producers & Energy Traders", 55105020: "Renewable Electricity",
    60101010: "Diversified REITs", 60101020: "Industrial REITs (2016-2023 code)",
    60101030: "Hotel & Resort REITs (2016-2023 code)", 60101040: "Office REITs (2016-2023 code)",
    60101050: "Health Care REITs (2016-2023 code)", 60101060: "Residential REITs (2016-2023 code)",
    60101070: "Retail REITs (2016-2023 code)", 60101080: "Specialized REITs (2016-2023 code)",
    60102010: "Diversified Real Estate Activities (2016-2023 code)",
    60102020: "Real Estate Operating Companies (2016-2023 code)",
    60102030: "Real Estate Development (2016-2023 code)", 60102040: "Real Estate Services (2016-2023 code)",
    60102510: "Industrial REITs", 60103010: "Hotel & Resort REITs", 60104010: "Office REITs",
    60105010: "Health Care REITs", 60106010: "Multi-Family Residential REITs",
    60106020: "Single-Family Residential REITs", 60107010: "Retail REITs", 60108010: "Other Specialized REITs",
    60108020: "Self-Storage REITs", 60108030: "Telecom Tower REITs", 60108040: "Timber REITs",
    60108050: "Data Center REITs", 60201010: "Diversified Real Estate Activities",
    60201020: "Real Estate Operating Companies", 60201030: "Real Estate Development",
    60201040: "Real Estate Services",
}
# Where the GICS codes come from (decision 39): no historical GICS is available, so every row carries the company's
# current (or last) classification from the Compustat header
GICS_SOURCE = "current_header"

# Near-zero denominators (s03): below these the ratio is missing
MIN_TTM_SALES_MUSD = 1.0            # f_sales_growth_1yr (prior TTM sales), f_rd_to_sales (current TTM sales)
MIN_TTM_CAPEX_MUSD = 1.0            # f_capex_growth_1yr (prior TTM capex)
MIN_SUE_STD = 0.01                  # f_sue: std of the previous 8 EPS changes, $ per share

# DCF (s04)
STATUTORY_TAX_RATE = {2017: 0.35}   # fiscal years up to 2017: 35%; from 2018 (Tax Cuts and Jobs Act): 21%
STATUTORY_TAX_RATE_FROM_2018 = 0.21
DCF_EXCLUDED_SECTORS = (40, 60)     # Financials and Real Estate: FCFF-based DCF does not apply

# IBES
IBES_TOTALS = ("SAL", "EBT", "EBI", "CPX")       # $ millions, no share basis
IBES_TOTAL_OUTLIER_MULTIPLE = 100               # a total above 100x both neighbouring estimates is NaN
IBES_MARGIN_MULTIPLE = 20                       # margin rule: EBIT/EBITDA (> 0) or capex above 20x TTM revenue ...
IBES_MARGIN_MIN_REVENUE_MUSD = 50.0             # ... when TTM revenue known at statpers is at least $50M
IBES_MARGIN_MEASURES = ("EBI", "EBT")          # IBES EBIT and EBITDA (only positive estimates); CPX any sign
IBES_OPEN_LINK_END = "2262-01-01"               # each permno's latest link is open-ended (WRDS ends active links)
IBES_LINK_SCORES = (1, 2)
IBES_CURRENCY = "USD"
IBES_MAX_AGE_DAYS = 60          # panel uses the latest statpers within 60 days of the rebalance date
PRICE_ASOF_DAYS = 7             # price on a date = last trading day on or before it, at most 7 days earlier

# Market closed (Hurricane Sandy) but Compustat has index levels; dropped so returns line up with CRSP
CLOSED_MARKET_DAYS = ("2012-10-29", "2012-10-30")

# s06 checks: sensible yield ranges (decimals), IWM total minus Russell 2000 price return per year (dividends minus
# fees), and how far a published monthly Fama-French factor may be from its compounded daily values
# (ust_10y up to 8%: 1997-2000 had 10-year yields near 7%)
YIELD_RANGES = {"ust_10y": (0.0, 0.08), "tbill_3m": (-0.005, 0.07), "moodys_baa": (0.02, 0.10),
                "moodys_aaa": (0.01, 0.09)}
IWM_MINUS_INDEX_RANGE = (0.005, 0.02)
FF_MONTHLY_TOLERANCE = 0.005
MAX_FILLED_RUN_DAYS = 5

# SIC fallback where Compustat GICS is missing (decision 4): the SIC code as of t mapped to a GICS sector name,
# checked in this order (first match wins); missing, 0 or 9900-9999 is Unclassified; any other code is Industrials
SIC_TO_GICS_SECTOR: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Health Care", ((2830, 2836), (3841, 3851), (8000, 8099), (8731, 8731))),
    ("Consumer Staples", ((100, 999), (2000, 2199), (2840, 2844), (5140, 5149), (5400, 5499), (5912, 5912))),
    ("Consumer Discretionary", ((2200, 2399), (2500, 2599), (3630, 3639), (3651, 3651), (3710, 3716),
                                (3750, 3751), (3940, 3949), (5200, 5999), (7000, 7099), (7200, 7299),
                                (7500, 7599), (7900, 7999), (8200, 8299))),
    ("Energy", ((1200, 1399), (2900, 2999))),
    ("Utilities", ((4900, 4949), (4960, 4999))),
    ("Real Estate", ((6500, 6599), (6798, 6798))),
    ("Financials", ((6000, 6799),)),
    ("Communication Services", ((2710, 2799), (4800, 4899), (7810, 7899))),
    ("Information Technology", ((3570, 3579), (3600, 3699), (3820, 3829), (7370, 7379))),
    ("Materials", ((1000, 1199), (1400, 1499), (2400, 2499), (2600, 2699), (2800, 2899), (3200, 3399))),
)
SIC_UNCLASSIFIED = "Unclassified"

# s07 panel
STALE_FUNDAMENTALS_DAYS = 365       # fundamentals older than this at t (t - period_end) are not used
SURPRISE_MAX_AGE_DAYS = 120         # analyst surprise from an announcement at most this old
SURPRISE_PRICE_MAX_DAYS = 10        # the price before an announcement is at most this many days older
BETA_DAYS, BETA_MONTHS = 252, 60    # beta windows; each needs MIN_OBS_SHARE of the observations
DISPERSION_MIN_MEAN = 0.01          # f_eps_dispersion needs |mean EPS estimate| >= $0.01 ...
DISPERSION_MIN_NUMEST = 2           # ... and at least 2 estimates
PEER_MIN = 5                        # percentile ranks and peer percentiles need 5 stocks with a value
SHARE_CLASS_GAP_COMPANY = 0.05      # Compustat shares above ours by more than 5% of the company count as a gap ...
SHARE_CLASS_GAP_YEAR = 0.01         # ... and the gaps may be at most 1% of the universe market cap in a year
COMPUSTAT_SHARES_MIN_EXCESS = 0.05  # company mkt cap from Compustat shares when they exceed ours by more than 5%
COMPUSTAT_FLAG_MIN_MONTHS = 3       # ... once the gap has held at each of the last 3 month ends, t included
RV_MIN_DENOMINATOR_MUSD = 1.0       # RV multiples: TTM and forward total denominators below $1M are NaN
RV_MIN_FORWARD_EPS = 0.01           # fwd_pe: forward EPS (CRSP basis) below $0.01 is NaN
FORWARD_YIELD_MAX_ABS = 5.0         # f_forward_ep, f_forward_ep_fy2, f_forward_cfp beyond +-5: NaN (counted)
SURPRISE_MAX_ABS = 1.0              # f_analyst_surprise beyond +-1: NaN (counted)
SANITY_MEDIAN_RANGES = {"pe": (15, 25), "pb": (1, 3), "ev_ebitda": (8, 15)}


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
