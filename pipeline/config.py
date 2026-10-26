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
    processed_dir: Path | None = None   # default <root>/processed; s09 reruns everything into a temporary one

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, processed_dir: str | Path | None = None) -> "DataPaths":
        root = Path(data_dir).expanduser().resolve()
        if not (root / "raw").is_dir():
            raise FileNotFoundError(f"{root} has no raw/ folder; pass the data folder with --data-dir")
        return cls(root, Path(processed_dir).expanduser().resolve() if processed_dir else None)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def query_inputs(self) -> Path:
        return self.raw / "query_inputs"

    @property
    def processed(self) -> Path:
        return self.processed_dir or self.root / "processed"

    def stage_dir(self, stage: str) -> Path:
        return self.processed / STAGE_FOLDERS.get(stage, stage)


STAGE_FOLDERS = {"s09_validate": "s09_validation"}       # stages whose folder is not named after them
OLD_PANEL_PATHS = ("archive/r2000_monthly_panel.parquet",  # the notebook's panel, for s09's old vs new backtest
                   "scratch/notebook_ref/r2000_monthly_panel.parquet")


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

# GICS names. Compustat's GICS is each company's latest classification (current for active companies, the last
# one used for inactive ones, so some discontinued industry codes appear). Industry names follow the 2023
# structure; discontinued codes keep their last name.
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
}

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
YIELD_RANGES = {"ust_10y": (0.0, 0.06), "tbill_3m": (-0.005, 0.07), "moodys_baa": (0.02, 0.10),
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
