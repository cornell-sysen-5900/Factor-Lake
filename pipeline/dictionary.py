"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: pipeline/dictionary.py
PURPOSE: The data dictionary of processed/final/: every file and every column with its meaning, unit, source and
         the stage that builds it. s08_export renders it to DATA_DICTIONARY.md and FAILs when an output column is
         missing here or a column here is missing from the outputs.

Each column is (meaning, unit, source[, stage]); the stage defaults to the file's stage.
"""

from __future__ import annotations

from pipeline.config import GICS_SECTORS, SIC_TO_GICS_SECTOR

# units
DEC, MUSD, PS, DATE, TEXT, FLAG, CODE, N, X, MSH = ("decimal", "$ millions", "$ per share", "date", "text", "true/false",
                                                     "code", "count", "ratio (times)", "millions of shares")

# sources
CIZ = "CRSP daily stock file (CIZ)"
IWM = "IWM holdings (CRSP mutual fund holdings)"
CQ = "Compustat quarterly (CCM)"
CA = "Compustat annual (CCM)"
IBES = "IBES summary statistics"
LINK = "CRSP-IBES link"
DERIVED = "derived"

ID = {
    "permno": ("CRSP security id (one share class)", CODE, CIZ),
    "permco": ("CRSP company id; share classes of one company share it", CODE, CIZ),
    "gvkey": ("Compustat company id", TEXT, CQ),
}
GICS = {
    "gsector": ("GICS sector code (2 digits); current classification, not historical", CODE, CQ),
    "gsector_name": ("GICS sector name", TEXT, "config.GICS_SECTORS"),
    "ggroup": ("GICS industry group code (4 digits)", CODE, CQ),
    "gind": ("GICS industry code (6 digits); the RV peer group", CODE, CQ),
    "gind_name": ("GICS industry name", TEXT, "config.GICS_INDUSTRIES"),
    "gsubind": ("GICS sub-industry code (8 digits)", CODE, CQ),
}

# ---------- statement factors and building blocks (s03, used in the panel) ----------

BLOCKS = {
    "book_equity": ("book equity: stockholders' equity (seqq, else ceqq + pstkq, else atq - ltq) + deferred taxes "
                    "(txditcq) - preferred (pstkq), Fama-French style", MUSD, CQ, "s03_fundamentals_q"),
    "ib_ttm": ("income before extraordinary items, sum of the last 4 fiscal quarters (ibq)", MUSD, CQ,
               "s03_fundamentals_q"),
    "revenue_ttm": ("net sales, last 4 fiscal quarters (saleq)", MUSD, CQ, "s03_fundamentals_q"),
    "dv_ttm": ("cash dividends paid, last 4 fiscal quarters (from year-to-date dvy)", MUSD, CQ, "s03_fundamentals_q"),
    "fcf_ttm": ("free cash flow, last 4 fiscal quarters: operating cash flow - capex", MUSD, CQ, "s03_fundamentals_q"),
    "ev_claims": ("claims added to market cap for EV: total debt - cash + preferred + minority interest", MUSD, CQ,
                  "s03_fundamentals_q"),
    "ebitda_ttm": ("operating income before depreciation, last 4 fiscal quarters (oibdpq)", MUSD, CQ,
                   "s03_fundamentals_q"),
    "ebit_ttm": ("operating income after depreciation, last 4 fiscal quarters (oiadpq)", MUSD, CQ,
                 "s03_fundamentals_q"),
    "cshoq": ("common shares outstanding at the fiscal quarter end, all classes", MSH, CQ, "s03_fundamentals_q"),
}
STATEMENT = {
    "f_roe": ("return on equity: TTM net income / book equity (book equity > 0)", DEC, CQ, "s03_fundamentals_q"),
    "f_roa": ("return on assets: TTM net income / total assets", DEC, CQ, "s03_fundamentals_q"),
    "f_gross_profitability": ("(TTM revenue - TTM cost of goods sold) / total assets", DEC, CQ, "s03_fundamentals_q"),
    "f_accruals_to_assets": ("(TTM net income - TTM operating cash flow) / total assets", DEC, CQ,
                             "s03_fundamentals_q"),
    "f_leverage": ("total debt / total assets", DEC, CQ, "s03_fundamentals_q"),
    "f_current_ratio": ("current assets / current liabilities (liabilities > 0)", X, CQ, "s03_fundamentals_q"),
    "f_rd_to_sales": ("TTM R&D (0 if not reported) / TTM revenue; NaN when TTM revenue < $1M", DEC, CQ,
                      "s03_fundamentals_q"),
    "f_asset_growth_1yr": ("total assets / total assets 4 fiscal quarters earlier - 1", DEC, CQ,
                           "s03_fundamentals_q"),
    "f_sales_growth_1yr": ("TTM revenue / TTM revenue 4 quarters earlier - 1; NaN when the earlier one < $1M", DEC,
                           CQ, "s03_fundamentals_q"),
    "f_capex_growth_1yr": ("TTM capex / TTM capex 4 quarters earlier - 1; NaN when the earlier one < $1M", DEC, CQ,
                           "s03_fundamentals_q"),
    "f_net_share_issuance": ("log(split-adjusted shares / split-adjusted shares 4 quarters earlier)", DEC, CQ,
                             "s03_fundamentals_q"),
    "f_sue": ("standardized unexpected earnings: change in split-adjusted EPS vs 4 quarters earlier / std of the "
              "previous 8 such changes (at least 6, std >= $0.01); NaN beyond +-50", "std units", CQ,
              "s03_fundamentals_q"),
}
QE_PRICE = {
    "f_book_to_price": "book equity / market cap at the quarter end (Compustat price x shares)",
    "f_earnings_to_price": "TTM split-adjusted diluted EPS / Compustat price at the quarter end",
    "f_sales_to_price": "TTM revenue / market cap at the quarter end",
    "f_ebitda_to_ev": "TTM EBITDA / (market cap at the quarter end + net debt)",
    "f_dividend_yield": "TTM split-adjusted dividends per share / Compustat price at the quarter end",
}

# ---------- files ----------

FILES: dict[str, dict] = {}

FILES["factor_panel_monthly.parquet"] = {
    "stage": "s07_panel", "row": "Russell 2000 member (IWM holding), month end t", "key": ["date", "permno"],
    "about": "The point-in-time factor panel for backtests: at each month end t, every stock in the latest IWM "
             "snapshot effective by t, with factors built only from data public by t. Factors are raw values; rank "
             "or z-score them per date yourself.",
    "columns": {
        "date": ("month end t: the last CRSP trading day of the month", DATE, CIZ),
        "permno": ID["permno"], "permco": ID["permco"], "gvkey": ("Compustat company id of the fundamentals row "
                                                                   "used at t", TEXT, CQ),
        "ticker": ("CRSP ticker at t (IWM ticker when CRSP has none)", TEXT, CIZ),
        "security_name": ("security name in the IWM snapshot", TEXT, IWM, "s01_membership"),
        **{k: v + ("s07_panel",) if len(v) == 3 else v for k, v in GICS.items()},
        "sector_name": ("GICS sector name, or where GICS is missing the SIC code mapped to a GICS sector name "
                        "(see the SIC fallback below)", TEXT, DERIVED),
        "gics_from_sic": ("sector_name comes from the SIC fallback", FLAG, DERIVED),
        "siccd": ("CRSP SIC code at t (last non-zero)", CODE, CIZ),
        "index_weight": ("weight in the IWM fund on the snapshot report date", "percent of the fund", IWM,
                         "s01_membership"),
        "snapshot_report_dt": ("report date of the IWM snapshot used at t", DATE, IWM, "s01_membership"),
        "snapshot_eff_dt": ("date the whole snapshot was public (latest eff_dt of its lines); always <= t", DATE,
                            IWM, "s01_membership"),
        "new_entrant": ("not in the universe at the previous month end", FLAG, DERIVED),
        "price_date": ("last real trading day of the stock in month t (never a delisting row)", DATE, CIZ,
                       "s02_prices"),
        "prc": ("closing price on price_date (|DlyPrc|; bid/ask midpoints made positive)", PS, CIZ, "s02_prices"),
        "shrout": ("shares outstanding on price_date", "thousands of shares", CIZ, "s02_prices"),
        "mkt_cap_musd": ("market cap of this listed security: prc x shrout. Use for portfolio weights", MUSD, CIZ,
                         "s02_prices"),
        "company_mkt_cap_musd": ("company market cap, for valuation (B/P, EV, multiples): all our permnos with the "
                                 "same permco, plus unlisted shares at the largest class's price when "
                                 "company_cap_from_compustat", MUSD, f"{CIZ}; {CQ}"),
        "company_mkt_cap_crsp_musd": ("company market cap from CRSP only: sum of mkt_cap_musd over our permnos with "
                                      "the same permco", MUSD, CIZ),
        "company_cap_from_compustat": ("Compustat common shares exceed our CRSP shares by more than 5% (an unlisted "
                                       "share class), so company_mkt_cap_musd uses the Compustat count. Never for "
                                       "ADRs; not for episodes of at most 3 months within 15% of CRSP's count (share "
                                       "issuance between quarter ends)", FLAG, DERIVED),
        "is_adr": ("CRSP share type is an ADR (security name '...; COM NONE ADR; ...'); a company with an ADR never "
                   "uses the Compustat share count", FLAG, CIZ, "s02_prices"),
        "n_share_classes": ("number of our permnos with this permco and a market cap at t", N, CIZ),
        "company_shares_crsp_m": ("shares of all our permnos with this permco at t", MSH, CIZ),
        "company_shares_compustat_m": ("Compustat cshoq at the quarter end put on the CRSP share basis at t "
                                       "(x cfacshr at the quarter end / cfacshr at t)", MSH, f"{CQ}; {CIZ}"),
        "mom_1m": ("price momentum: compounded return over the last 21 trading days to price_date (80% of days "
                   "needed)", DEC, CIZ, "s02_prices"),
        "mom_6m": ("compounded return over the last 126 trading days", DEC, CIZ, "s02_prices"),
        "mom_12m": ("compounded return over the last 252 trading days", DEC, CIZ, "s02_prices"),
        "mom_12_1": ("12-month momentum skipping the last month: (1 + mom_12m) / (1 + mom_1m) - 1", DEC, DERIVED),
        **{f"vol_{kind}_{w}": (f"{'Garman-Klass (open, high, low, close)' if kind == 'gk' else 'close-to-close'} "
                               f"volatility over the last {n} trading days, annualized", "annualized decimal", CIZ,
                               "s02_prices")
           for w, n in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)) for kind in ("gk", "close")},
        "beta_252d": ("market beta: cov / var of daily excess returns (stock - rf vs Fama-French mktrf) over the last "
                      "252 trading rows, at least 202", X, f"{CIZ}; Fama-French"),
        "beta_60m": ("market beta from monthly excess returns over the last 60 months, at least 48", X,
                     f"{CIZ}; Fama-French"),
        "fund_period_end": ("fiscal quarter end of the fundamentals used at t", DATE, CQ, "s03_fundamentals_q"),
        "fund_report_date": ("date those fundamentals were public (rdq, else period end + 45 days, Q4 + 90); "
                             "always <= t", DATE, CQ, "s03_fundamentals_q"),
        "fund_is_usd": ("the statements are in USD (price-based factors are NaN otherwise)", FLAG, CQ,
                        "s03_fundamentals_q"),
        "fund_stale": ("the fundamentals are more than 365 days old at t: every fundamental column is NaN", FLAG,
                       DERIVED),
        **BLOCKS,
        "ev": ("enterprise value at t: company_mkt_cap_musd + ev_claims; NaN when <= 0", MUSD, DERIVED),
        **STATEMENT,
        "f_book_to_price": ("book equity / company market cap at t (book equity > 0)", DEC, DERIVED),
        "f_earnings_to_price": ("TTM income before extraordinary items / company market cap at t", DEC, DERIVED),
        "f_sales_to_price": ("TTM revenue / company market cap at t", DEC, DERIVED),
        "f_dividend_yield": ("TTM dividends / company market cap at t", DEC, DERIVED),
        "f_fcf_yield": ("TTM free cash flow / company market cap at t", DEC, DERIVED),
        "f_ebitda_to_ev": ("TTM EBITDA / EV at t", DEC, DERIVED),
        "f_ebit_to_ev": ("TTM EBIT / EV at t", DEC, DERIVED),
        "f_sales_to_ev": ("TTM revenue / EV at t", DEC, DERIVED),
        **{f"{k}_qe": (f"{v}: the notebook's quarter-end version, from Compustat's price (NaN for non-USD and "
                       "prices below $0.01)", DEC, CQ, "s03_fundamentals_q") for k, v in QE_PRICE.items()},
        "ibes_statpers": ("IBES statistics date of the EPS FY1 consensus used at t (<= t, at most 60 days old)",
                          DATE, IBES, "s05_ibes"),
        "numest_fy1": ("number of EPS FY1 estimates in that consensus", N, IBES, "s05_ibes"),
        "f_forward_ep": ("forward E/P: EPS FY1 consensus on the CRSP share basis at t / prc; NaN beyond +-5", DEC,
                         f"{IBES}; {CIZ}"),
        "f_forward_ep_fy2": ("EPS FY2 consensus on the CRSP share basis at t / prc; NaN beyond +-5", DEC,
                             f"{IBES}; {CIZ}"),
        "f_forward_cfp": ("cash flow per share FY1 consensus on the CRSP share basis at t / prc; NaN beyond +-5",
                          DEC, f"{IBES}; {CIZ}"),
        "f_forward_sales_to_ev": ("sales FY1 consensus / EV at t", DEC, IBES),
        "f_forward_ebitda_to_ev": ("EBITDA FY1 consensus (IBES measure EBT) / EV at t", DEC, IBES),
        "f_ltg": ("long-term EPS growth consensus (IBES FPI 0)", "decimal per year", IBES),
        "f_eps_revision": ("(estimates revised up - revised down) / number of estimates, EPS FY1", DEC, IBES),
        "f_eps_dispersion": ("std of EPS FY1 estimates / |mean|; NaN when |mean| < $0.01 or fewer than 2 "
                             "estimates", X, IBES),
        "f_analyst_surprise": ("latest EPS surprise announced on or before t (at most 120 days old): (actual - last "
                               "consensus before the announcement) / price on the last trading day before it, both "
                               "on that day's CRSP share basis; NaN beyond +-1", DEC, f"{IBES}; {CIZ}"),
        "surprise_anndats": ("announcement date of that surprise", DATE, IBES, "s05_ibes"),
        "surprise_fpedats": ("fiscal period end of that surprise", DATE, IBES, "s05_ibes"),
        "ret_next_1m": ("the stock's return in month t+1, delisting return included (the backtest's holding "
                        "return; never use it as a factor)", DEC, CIZ, "s02_prices"),
        "delisted_next_month": ("the stock delists in month t+1", FLAG, CIZ, "s02_prices"),
        "eps2_statpers": ("statistics date of the EPS FY2 consensus used at t", DATE, IBES, "s05_ibes"),
        "cps1_statpers": ("statistics date of the CPS FY1 consensus used at t", DATE, IBES, "s05_ibes"),
        "sal1_statpers": ("statistics date of the SAL FY1 consensus used at t", DATE, IBES, "s05_ibes"),
        "ebt1_statpers": ("statistics date of the EBT (EBITDA) FY1 consensus used at t", DATE, IBES, "s05_ibes"),
        "ltg_statpers": ("statistics date of the long-term growth consensus used at t", DATE, IBES, "s05_ibes"),
    },
}

FILES["universe_periods.parquet"] = {
    "stage": "s01_membership", "row": "stock, membership period", "key": ["permno", "index_effective"],
    "about": "Periods each stock was in the IWM snapshots (consecutive snapshots, one missing snapshot bridged).",
    "columns": {
        "permno": ID["permno"],
        "index_effective": ("report date of the period's first snapshot", DATE, IWM),
        "index_through": ("report date of the period's last snapshot", DATE, IWM),
        "first_known": ("first date the period was public (earliest eff_dt)", DATE, IWM),
        "n_snapshots": ("snapshots in the period", N, IWM),
        "n_excluded": ("snapshots of the period excluded as non-members (decision 11)", N, DERIVED),
        "excluded_non_member": ("every snapshot of the period is excluded: a merger stock that was never really a "
                                "member, not in the panel", FLAG, DERIVED),
        "ticker": ("IWM ticker in the period's last snapshot", TEXT, IWM),
        "security_name": ("IWM security name in the period's last snapshot", TEXT, IWM),
    },
}

FILES["returns_monthly.parquet"] = {
    "stage": "s02_prices", "row": "stock, month (every permno ever in the universe, also after it leaves)",
    "key": ["month_end", "permno"],
    "about": "Monthly returns for backtest holding periods, including the months after a stock leaves the index "
             "and the delisting month.",
    "columns": {
        "month_end": ("last CRSP trading day of the month", DATE, CIZ),
        "permno": ID["permno"], "permco": ID["permco"],
        "ticker": ("CRSP ticker, last in the month", TEXT, CIZ),
        "last_trade_date": ("last real trading day of the stock in the month", DATE, CIZ),
        "prc": ("closing price on last_trade_date", PS, CIZ),
        "shrout": ("shares outstanding on last_trade_date", "thousands of shares", CIZ),
        "mkt_cap_musd": ("prc x shrout on last_trade_date", MUSD, CIZ),
        "ret_m": ("monthly return: every daily return of the month compounded, the delisting return included", DEC,
                  CIZ),
        "trading_days": ("real trading days of the stock in the month", N, CIZ),
        "delisted_in_month": ("the month has the stock's delisting row", FLAG, CIZ),
        "delist_ret_filled": ("the delisting return was missing and filled (-30% for GDR, else 0%)", FLAG, DERIVED),
        "is_adr": ("CRSP share type is an ADR (from the security name)", FLAG, CIZ),
    },
}

FILES["prices_daily/year=YYYY/data.parquet"] = {
    "stage": "s02_prices", "row": "stock, trading day (one file per calendar year)", "key": ["permno", "date"],
    "about": "Daily CRSP prices of every permno ever in the universe, in the layout of the repo's "
             "scripts/publish_daily_data_to_s3.py (one file per year, sorted by permno and date) plus "
             "is_delisting_row. Momentum and volatility are in the monthly panel, not here.",
    "columns": {
        "permno": ID["permno"],
        "ticker": ("CRSP ticker", TEXT, CIZ),
        "date": ("trading day", DATE, CIZ),
        "prc": ("closing price, |DlyPrc|; NaN on delisting rows and when CRSP has no price", PS, CIZ),
        "prc_adj": ("split-adjusted price: prc / cfacpr (prc when cfacpr is missing or 0)", PS, CIZ),
        "cfacpr": ("CRSP cumulative price adjustment factor (DlyCumFacPr); falls at splits, 1 at the end of the "
                   "data", X, CIZ),
        "shrout": ("shares outstanding", "thousands of shares", CIZ),
        "mkt_cap_musd": ("prc x shrout", MUSD, CIZ),
        "ret": ("daily return (DlyRet); on the delisting row the delisting return (filled -30% for GDR, else 0% "
                "when missing)", DEC, CIZ),
        "vol": ("shares traded (DlyVol)", "shares", CIZ),
        "is_delisting_row": ("CRSP's delisting row (DlyDelFlg = Y): carries only the delisting return, no price",
                             FLAG, CIZ),
    },
}

_MARKET = {
    "r2000_price_ret": ("Russell 2000 index price return (no dividends)", DEC, "Compustat index prices (RUT)"),
    "r2000_growth_price_ret": ("Russell 2000 Growth index price return", DEC, "Compustat index prices"),
    "r2000_value_price_ret": ("Russell 2000 Value index price return", DEC, "Compustat index prices"),
    "iwm_total_ret": ("IWM ETF total return (dividends included)", DEC, CIZ),
    "iwn_total_ret": ("IWN (Russell 2000 Value ETF) total return", DEC, CIZ),
    "iwo_total_ret": ("IWO (Russell 2000 Growth ETF) total return", DEC, CIZ),
    "mktrf": ("Fama-French market excess return", DEC, "Fama-French (WRDS)"),
    "smb": ("Fama-French size factor (small minus big)", DEC, "Fama-French (WRDS)"),
    "hml": ("Fama-French value factor (high minus low book-to-market)", DEC, "Fama-French (WRDS)"),
    "rmw": ("Fama-French profitability factor (robust minus weak)", DEC, "Fama-French (WRDS)"),
    "cma": ("Fama-French investment factor (conservative minus aggressive)", DEC, "Fama-French (WRDS)"),
    "umd": ("momentum factor (up minus down)", DEC, "Fama-French (WRDS)"),
    "rf": ("risk-free rate for the period (one-month T-bill)", DEC, "Fama-French (WRDS)"),
    "ust_10y": ("10-year Treasury yield", "decimal per year", "FRED DGS10"),
    "tbill_3m": ("3-month Treasury bill yield", "decimal per year", "FRED DGS3MO"),
    "moodys_baa": ("Moody's Baa corporate bond yield (the DCF cost-of-debt fallback)", "decimal per year",
                   "FRED DBAA"),
    "moodys_aaa": ("Moody's Aaa corporate bond yield", "decimal per year", "FRED DAAA"),
    "term_spread": ("ust_10y - tbill_3m", "decimal per year", DERIVED),
    "credit_spread": ("moodys_baa - moodys_aaa", "decimal per year", DERIVED),
}
FILES["market_daily.parquet"] = {
    "stage": "s06_market", "row": "trading day", "key": ["date"],
    "about": "Benchmarks, Fama-French factors and interest rates by day. Returns are daily; yields are annual rates "
             "on that day (holidays forward-filled).",
    "columns": {"date": ("trading day", DATE, DERIVED), **_MARKET,
                "was_filled": ("at least one yield was forward-filled from an earlier day", FLAG, DERIVED)},
}
FILES["market_monthly.parquet"] = {
    "stage": "s06_market", "row": "month", "key": ["month"],
    "about": "The same by month: index returns from month-end levels, ETF returns compounded from daily, "
             "Fama-French as published monthly, yields and spreads as of the month end.",
    "columns": {"month": ("month as YYYY-MM", TEXT, DERIVED),
                "month_end": ("last CRSP trading day of the month", DATE, CIZ),
                **{k: (v[0].replace("return", "monthly return", 1) if v[1] == DEC else v[0] + " at the month end",
                       *v[1:]) for k, v in _MARKET.items()}},
}

RV_MULTIPLES = {
    "pe": ("price / earnings: company market cap / TTM income before extraordinary items (>= $1M)", X),
    "pb": ("price / book: company market cap / book equity (> 0)", X),
    "ps": ("price / sales: company market cap / TTM revenue (>= $1M)", X),
    "ev_ebitda": ("EV / TTM EBITDA (>= $1M)", X),
    "ev_ebit": ("EV / TTM EBIT (>= $1M)", X),
    "ev_sales": ("EV / TTM revenue (>= $1M)", X),
    "fcf_yield": ("TTM free cash flow / company market cap (may be negative)", DEC),
    "dividend_yield": ("TTM dividends / company market cap", DEC),
    "earnings_yield": ("TTM income before extraordinary items / company market cap (may be negative)", DEC),
    "fwd_pe_fy1": ("prc / EPS FY1 consensus on the CRSP share basis at t (EPS >= $0.01)", X),
    "fwd_pe_fy2": ("prc / EPS FY2 consensus on the CRSP share basis at t (EPS >= $0.01)", X),
    "fwd_ev_ebitda": ("EV / EBITDA FY1 consensus (>= $1M)", X),
    "fwd_ev_sales": ("EV / sales FY1 consensus (>= $1M)", X),
    "ltg": ("long-term EPS growth consensus (IBES FPI 0)", "decimal per year"),
    "peg": ("fwd_pe_fy1 / (ltg x 100), only when ltg > 0", X),
}
FILES["rv_multiples_monthly.parquet"] = {
    "stage": "s07_panel", "row": "Russell 2000 member, month end t (the panel's rows)", "key": ["date", "permno"],
    "about": "Valuation multiples for the relative-value screen, with each stock's percentile rank within its GICS "
             "industry and sector at t. Multiples are NaN when the denominator is below its floor; ranks are NaN "
             "when fewer than 5 stocks of the group have a value. NaN for non-USD statements and prices below $0.01.",
    "columns": {
        "date": ("month end t", DATE, CIZ), "permno": ID["permno"],
        "gsector": GICS["gsector"], "gind": GICS["gind"],
        "prc": ("closing price at t", PS, CIZ, "s02_prices"),
        "company_mkt_cap_musd": ("company market cap at t (see factor_panel_monthly)", MUSD, f"{CIZ}; {CQ}"),
        "company_cap_from_compustat": ("company_mkt_cap_musd uses the Compustat share count", FLAG, DERIVED),
        "ev": ("enterprise value at t", MUSD, DERIVED),
        **{k: (m, u, DERIVED) for k, (m, u) in RV_MULTIPLES.items()},
        **{f"{k}_pct_{g}": (f"percentile rank (0-1, highest = 1) of {k} within the GICS {name} at t", "0-1",
                            DERIVED) for g, name in (("gind", "industry"), ("gsector", "sector"))
           for k in RV_MULTIPLES},
    },
}
FILES["rv_peer_stats_monthly.parquet"] = {
    "stage": "s07_panel", "row": "GICS industry, month end t", "key": ["date", "gind"],
    "about": "Peer distribution of every multiple within each GICS industry at t (percentiles NaN with fewer than 5 "
             "values).",
    "columns": {
        "date": ("month end t", DATE, CIZ), "gind": GICS["gind"], "gind_name": GICS["gind_name"],
        "n_peers": ("universe stocks in the industry at t", N, DERIVED),
        **{f"{k}_{s}": ((f"stocks with a {k} value" if s == "n" else
                         f"{ {'p25': '25th', 'p50': '50th (median)', 'p75': '75th'}[s]} percentile of {k}"),
                        N if s == "n" else u, DERIVED)
           for k, (_, u) in RV_MULTIPLES.items() for s in ("n", "p25", "p50", "p75")},
    },
}

FILES["dcf_inputs_annual.parquet"] = {
    "stage": "s04_fundamentals_a", "row": "stock, fiscal year", "key": ["permno", "fyear"],
    "about": "Annual statements for DCF valuation. Use a row only from report_date on (point in time). Financials "
             "and Real Estate are kept but marked dcf_applicable = False.",
    "columns": {
        "permno": ID["permno"], "gvkey": ID["gvkey"][:2] + (CA,),
        "tic": ("Compustat ticker", TEXT, CA), "conm": ("Compustat company name", TEXT, CA),
        "fyear": ("fiscal year", "year", CA), "fyr": ("fiscal year-end month", "month (1-12)", CA),
        "period_end": ("fiscal year end (datadate)", DATE, CA),
        "report_date": ("date the annual numbers were public: rdq of the quarterly Q4 row with the same period end, "
                        "else period end + 90 days", DATE, f"{CA}; {CQ}"),
        "report_date_source": ("q4_rdq or proxy (period end + 90 days)", TEXT, DERIVED),
        "curcd": ("reporting currency", TEXT, CA), "is_usd": ("reported in USD", FLAG, CA),
        "ajex": ("Compustat cumulative adjustment factor (divide per-share items, multiply shares)", X, CA),
        "price_fye": ("closing price at the fiscal year end (prcc_f)", PS, CA),
        "epsfx": ("diluted EPS excluding extraordinary items", PS, CA),
        "emp": ("employees", "thousands", CA),
        "sich": ("historical SIC code", CODE, CA),
        **{k: v[:2] + (CA,) for k, v in GICS.items()},
        "naics": ("NAICS industry code", CODE, CA),
        "dcf_applicable": ("False for GICS sectors 40 (Financials) and 60 (Real Estate), where a DCF on FCFF does "
                           "not fit", FLAG, DERIVED),
        "interest_source": ("where interest_expense comes from: xint, xintd or missing (never 0-filled; the DCF "
                            "then uses the Baa yield)", TEXT, DERIVED),
        "revenue": ("net sales (sale)", MUSD, CA), "ebitda": ("operating income before depreciation (oibdp)", MUSD, CA),
        "ebit": ("operating income after depreciation (oiadp)", MUSD, CA),
        "da": ("depreciation and amortization (dp)", MUSD, CA), "capex": ("capital expenditures (capx)", MUSD, CA),
        "oancf": ("operating cash flow", MUSD, CA), "pretax_income": ("pretax income (pi)", MUSD, CA),
        "taxes": ("income taxes (txt)", MUSD, CA),
        "tax_rate": ("effective tax rate: taxes / pretax income when pretax > 0, clipped to 0-60%", DEC, DERIVED),
        "tax_rate_fcff": ("tax rate for FCFF: tax_rate, or 0 when pretax income <= 0", DEC, DERIVED),
        "statutory_tax_rate": ("US federal statutory rate: 35% up to fiscal 2017, 21% from 2018", DEC,
                               "config.STATUTORY_TAX_RATE"),
        "interest_expense": ("interest expense: xint, else xintd", MUSD, CA),
        "net_income": ("net income (ni)", MUSD, CA),
        "nwc": ("non-cash working capital: (current assets - cash) - (current liabilities - short-term debt)", MUSD,
                CA),
        "d_nwc": ("change in nwc vs fiscal year - 1 (exactly the previous year)", MUSD, DERIVED),
        "total_debt": ("long-term debt + debt in current liabilities", MUSD, CA), "cash": ("cash and short-term "
                                                                                            "investments (che)",
                                                                                            MUSD, CA),
        "net_debt": ("total_debt - cash", MUSD, DERIVED), "preferred": ("preferred stock (pstk)", MUSD, CA),
        "minority_interest": ("noncontrolling interest (mib)", MUSD, CA),
        "diluted_shares": ("shares for diluted EPS (cshfd)", MSH, CA),
        "basic_shares": ("common shares outstanding (csho)", MSH, CA),
        "book_equity": ("book equity, Fama-French style (as in fundamentals_quarterly)", MUSD, CA),
        "dividends": ("common dividends (dvc)", MUSD, CA), "dividends_total": ("total cash dividends (dv)", MUSD, CA),
        "buybacks": ("purchase of common and preferred stock (prstkc)", MUSD, CA),
        "share_issuance": ("sale of common and preferred stock (sstk)", MUSD, CA),
        "fcff": ("free cash flow to the firm: EBIT x (1 - tax_rate_fcff) + D&A - capex - d_nwc; NaN if a part is "
                 "missing", MUSD, DERIVED),
        "revenue_growth": ("revenue / revenue of fiscal year - 1 - 1 (previous > 0)", DEC, DERIVED),
        "ebitda_growth": ("EBITDA / EBITDA of fiscal year - 1 - 1 (previous > 0)", DEC, DERIVED),
    },
}

FILES["estimates_consensus.parquet"] = {
    "stage": "s05_ibes", "row": "stock, IBES statistics date, measure, forecast period",
    "key": ["permno", "statpers", "measure", "fpi"],
    "about": "The full monthly IBES consensus in USD for 7 measures, linked to permno. IBES per-share values are "
             "on the latest share basis: multiply by crsp_basis_factor to compare with the CRSP price at statpers. "
             "Use a row only from statpers on.",
    "columns": {
        "permno": ID["permno"], "ibes_ticker": ("IBES ticker", TEXT, IBES),
        "statpers": ("IBES statistics date (when the consensus was computed)", DATE, IBES),
        "measure": ("EPS, CPS (cash flow per share), DPS (dividends per share): $ per share; SAL (sales), EBT "
                    "(EBITDA), EBI (EBIT), CPX (capex): $ millions", TEXT, IBES),
        "fpi": ("forecast period: 0 long-term growth, 1 FY1, 2 FY2, 3 FY3", CODE, IBES),
        "fpedats": ("fiscal period end being forecast", DATE, IBES),
        "numest": ("number of estimates", N, IBES), "numup": ("estimates revised up in the last month", N, IBES),
        "numdown": ("estimates revised down in the last month", N, IBES),
        "meanest": ("mean estimate (per share, $ millions, or decimal growth for FPI 0)", "see measure", IBES),
        "medest": ("median estimate", "see measure", IBES), "stdev": ("std of the estimates", "see measure", IBES),
        "highest": ("highest estimate", "see measure", IBES), "lowest": ("lowest estimate", "see measure", IBES),
        "actual": ("reported actual for fpedats (NaN when IBES has two different actuals)", "see measure", IBES),
        "anndats_act": ("announcement date of the actual", DATE, IBES),
        "price_statpers": ("CRSP price on statpers (last trading day, at most 7 days earlier)", PS, CIZ),
        "cfacpr_statpers": ("CRSP cfacpr on that day", X, CIZ),
        "crsp_basis_factor": ("cfacpr_statpers / the permno's last cfacpr (1 in CIZ, so equal to "
                              "cfacpr_statpers): per-share value x this = on the CRSP basis at "
                              "statpers", X, DERIVED),
        "suspect_total": ("a total (SAL, EBT, EBI, CPX) more than 100x both its previous and next estimate; its "
                          "values are NaN", FLAG, DERIVED),
        "suspect_scale": ("a total estimate or actual more than 100x the largest of the company's Compustat TTM "
                          "revenue, total assets and market cap known at statpers (scale rule), or, with TTM revenue "
                          ">= $50M, a positive EBIT/EBITDA or a capex estimate above 20x that revenue (margin rule); "
                          "those values are NaN", FLAG, DERIVED),
        "suspect_scale_reason": ("which rule set suspect_scale: scale or margin (scale when both); empty otherwise",
                                 TEXT, DERIVED),
    },
}

_CQ_ITEMS = {
    "atq": "total assets", "ltq": "total liabilities", "actq": "current assets", "lctq": "current liabilities",
    "cheq": "cash and short-term investments", "invtq": "inventories", "rectq": "receivables",
    "apq": "accounts payable", "ppentq": "net property, plant and equipment", "gdwlq": "goodwill",
    "intanq": "intangible assets", "dlcq": "debt in current liabilities", "dlttq": "long-term debt",
    "pstkq": "preferred stock", "mibtq": "noncontrolling interest", "seqq": "stockholders' equity",
    "ceqq": "common equity", "txditcq": "deferred taxes and investment tax credit", "saleq": "net sales",
    "revtq": "total revenue", "cogsq": "cost of goods sold", "xsgaq": "selling, general and administrative expense",
    "xrdq": "R&D expense", "oibdpq": "operating income before depreciation (EBITDA)", "dpq": "depreciation and "
    "amortization", "oiadpq": "operating income after depreciation (EBIT)", "xintq": "interest expense",
    "piq": "pretax income", "txtq": "income taxes", "niq": "net income", "ibq": "income before extraordinary items",
}
_CQ_PS = {"epsfxq": "diluted EPS excluding extraordinary items", "epspxq": "basic EPS excluding extraordinary items",
          "dvpsxq": "dividends per share by ex-date"}
_CF = {"oancf": "operating cash flow", "capx": "capital expenditures", "dv": "cash dividends",
       "sstk": "sale of common and preferred stock", "prstkc": "purchase of common and preferred stock",
       "dltis": "long-term debt issued", "dltr": "long-term debt repaid", "aqc": "acquisitions"}
_TTM = {"revenue": "saleq", "revt": "revtq", "cogs": "cogsq", "xsga": "xsgaq", "xrd": "xrdq", "ebitda": "oibdpq",
        "ebit": "oiadpq", "da": "dpq", "interest": "xintq", "pretax": "piq", "tax": "txtq", "ni": "niq", "ib": "ibq",
        "oancf": "oancf_q", "capx": "capx_q", "dv": "dv_q", "sstk": "sstk_q", "prstkc": "prstkc_q", "dltis": "dltis_q",
        "dltr": "dltr_q", "aqc": "aqc_q"}
FILES["fundamentals_quarterly.parquet"] = {
    "stage": "s03_fundamentals_q", "row": "stock, fiscal quarter", "key": ["permno", "fyearq", "fqtr"],
    "about": "Quarterly Compustat statements with TTM sums, building blocks and the quarter-end factors. TTM = "
             "exactly 4 consecutive fiscal quarters (NaN if one is missing). Use a row only from report_date on. "
             "After a fiscal-year change one period end can appear twice (old and new fiscal numbering).",
    "columns": {
        "permno": ID["permno"], "gvkey": ID["gvkey"],
        "tic": ("Compustat ticker", TEXT, CQ), "conm": ("Compustat company name", TEXT, CQ),
        "period_end": ("fiscal quarter end (datadate)", DATE, CQ),
        "report_date": ("date the quarter was public: rdq, else period end + 45 days (+ 90 for fiscal Q4)", DATE,
                        CQ),
        "report_date_source": ("rdq or proxy", TEXT, DERIVED), "rdq": ("Compustat report date", DATE, CQ),
        "fyearq": ("fiscal year", "year", CQ), "fqtr": ("fiscal quarter (1-4)", CODE, CQ),
        "curcdq": ("reporting currency", TEXT, CQ), "is_usd": ("reported in USD", FLAG, CQ),
        "ajexq": ("Compustat cumulative adjustment factor (divide per-share items, multiply shares)", X, CQ),
        **{k: v[:2] + (CQ,) for k, v in GICS.items()},
        "naics": ("NAICS industry code", CODE, CQ),
        "price_qe": ("closing price at the quarter end (prccq)", PS, CQ),
        "cshoq": ("common shares outstanding, all classes", MSH, CQ),
        **{k: (v, MUSD, CQ) for k, v in _CQ_ITEMS.items()},
        **{k: (v, PS, CQ) for k, v in _CQ_PS.items()},
        **{f"{k}_q": (f"{v} in the quarter (from Compustat's year-to-date {k}y: Q1 = YTD, later = YTD - previous "
                      "quarter's YTD)", MUSD, CQ) for k, v in _CF.items()},
        "book_equity": BLOCKS["book_equity"][:3],
        "total_debt": ("long-term debt + debt in current liabilities", MUSD, CQ),
        "cash": ("cash and short-term investments (cheq)", MUSD, CQ),
        "net_debt": ("total_debt - cash", MUSD, DERIVED), "preferred": ("preferred stock (pstkq)", MUSD, CQ),
        "minority_interest": ("noncontrolling interest (mibtq)", MUSD, CQ),
        "ev_claims": BLOCKS["ev_claims"][:3],
        "nwc": ("non-cash working capital: (current assets - cash) - (current liabilities - short-term debt)", MUSD,
                CQ),
        "d_nwc_4q": ("nwc - nwc 4 fiscal quarters earlier", MUSD, DERIVED),
        **{f"{k}_ttm": (f"{_CQ_ITEMS.get(v, _CF.get(v[:-2], v))}, sum of the last 4 fiscal quarters ({v})", MUSD,
                        DERIVED) for k, v in _TTM.items()},
        "fcf_ttm": ("oancf_ttm - capx_ttm", MUSD, DERIVED),
        "tax_rate_ttm": ("tax_ttm / pretax_ttm when pretax > 0, clipped to 0-60%", DEC, DERIVED),
        "epsfxq_ttm_adj": ("TTM diluted EPS, each quarter split-adjusted with ajexq, in this quarter's shares", PS,
                           DERIVED),
        "dvpsxq_ttm_adj": ("TTM dividends per share, split-adjusted the same way", PS, DERIVED),
        "diluted_shares": ("shares for diluted EPS (cshfdq)", MSH, CQ),
        "csho_adj": ("split-adjusted shares outstanding: cshoq x ajexq (comparable over time)", "millions of shares "
                     "(adjusted)", DERIVED),
        "cshfd_adj": ("split-adjusted diluted shares: cshfdq x ajexq", "millions of shares (adjusted)", DERIVED),
        "mkt_cap_qe_musd": ("market cap at the quarter end: price_qe x cshoq", MUSD, DERIVED),
        **{k: (f"{v} (NaN for non-USD and prices below $0.01)", DEC, DERIVED) for k, v in QE_PRICE.items()},
        **{k: v[:3] for k, v in STATEMENT.items()},
    },
}

FILES["security_master.parquet"] = {
    "stage": "s07_panel", "row": "permno ever in the universe (permno_list.txt)", "key": ["permno"],
    "about": "Lookup table: the latest ids, names and classification of every permno.",
    "columns": {
        "permno": ID["permno"], "permco": ("latest CRSP company id", CODE, CIZ),
        "gvkey": ("latest Compustat company id", TEXT, CQ),
        "ibes_ticker": ("every linked IBES ticker, ';'-separated, oldest first", TEXT, LINK),
        "ticker": ("latest CRSP ticker", TEXT, CIZ),
        "security_name": ("latest IWM security name", TEXT, IWM),
        **{k: (v[0].replace("current classification, not historical", "latest") if k == "gsector" else v[0],
               *v[1:]) for k, v in GICS.items()},
        "siccd": ("latest CRSP SIC code", CODE, CIZ),
        "first_price_date": ("first CRSP trading day in our data", DATE, CIZ),
        "last_price_date": ("last CRSP trading day in our data", DATE, CIZ),
        "first_universe_date": ("first month end in factor_panel_monthly", DATE, DERIVED),
        "last_universe_date": ("last month end in factor_panel_monthly", DATE, DERIVED),
        "excluded_non_member": ("at least one of its IWM snapshots was excluded as a non-member (decision 11)", FLAG,
                                DERIVED),
    },
}


FILES["column_stats.parquet"] = {
    "stage": "s08_export", "row": "final table, column", "key": ["file", "column"],
    "about": "Statistics of every column of every other final table (prices_daily's years combined), rebuilt on "
             "every export; the Stats tables in this dictionary are printed from it.",
    "columns": {
        "file": ("final table (prices_daily/year=YYYY/data.parquet for all daily years)", TEXT, DERIVED),
        "column": ("column name", TEXT, DERIVED),
        "kind": ("numeric, date, text or bool", TEXT, DERIVED),
        "dtype": ("Parquet (Arrow) type", TEXT, DERIVED),
        "rows": ("rows of the table", N, DERIVED),
        "non_null_pct": ("share of rows with a value", "percent", DERIVED),
        **{s: (f"{name} of the non-missing values (numeric columns)", "the column's unit", DERIVED)
           for s, name in (("min", "minimum"), ("p1", "1st percentile"), ("p25", "25th percentile"),
                           ("median", "median"), ("mean", "mean"), ("p75", "75th percentile"),
                           ("p99", "99th percentile"), ("max", "maximum"))},
        "date_min": ("earliest date (date columns)", DATE, DERIVED),
        "date_max": ("latest date (date columns)", DATE, DERIVED),
        "unique": ("distinct values (text and date columns)", N, DERIVED),
        "top_values": ("the 5 most common values with their share of rows, as JSON [[value, percent], ...] (text "
                       "columns)", TEXT, DERIVED),
        "true_pct": ("share True of the non-missing values (boolean columns)", "percent", DERIVED),
    },
}


def columns(name: str) -> dict[str, tuple[str, str, str, str]]:
    """The file's columns as (meaning, unit, source, stage)."""
    f = FILES[name]
    return {c: (*v[:3], v[3] if len(v) > 3 else f["stage"]) for c, v in f["columns"].items()}


# ---------- rendering ----------

def render(manifest: dict, stats=None, coverage=None) -> str:
    """DATA_DICTIONARY.md for teammates who never saw the pipeline; types and row counts from the manifest, a Stats
    table per file from column_stats (stats), and factor coverage by year for factor_panel_monthly."""
    from pipeline import column_stats
    out = manifest["outputs"]
    md = [f"# Factor-Lake WRDS data: data dictionary (version {manifest['data_version']})", "",
          "Built by the WRDS pipeline in the Factor-Lake repo (`pipeline/`, commit "
          f"`{manifest['pipeline_git_commit'][:12]}`) from the WRDS, Fama-French and FRED downloads listed in "
          "manifest.json. All files are Parquet; read them with `pandas.read_parquet`. The schema-only version is "
          "`pipeline/DATA_DICTIONARY.md` in the repo; this one adds the stats of the data.", "",
          *CONVENTIONS,
          "## Files", "", "| file | one row per | key | rows | MB |", "|---|---|---|---|---|"]
    for name, f in FILES.items():
        o = out[name]
        md.append(f"| [{name}](#{_anchor(name)}) | {f['row']} | {', '.join(f['key'])} | {o['rows']:,} | "
                  f"{o['size_mb']:,.1f} |")
    md += ["", "Also: `prices_daily/_manifest.json` (years, rows, sizes of the daily files), `manifest.json` "
           "(sources, checksums, units, known gaps) and `UPLOAD.md`.", ""]
    for name, f in FILES.items():
        o = out[name]
        rng = f", {o['date_range'][0]} to {o['date_range'][1]}" if o.get("date_range") else ""
        md += [f"## {name}", "", f["about"], "",
               f"One row per {f['row']}; key {', '.join(f['key'])}. Built by stage `{f['stage']}`. "
               f"{o['rows']:,} rows{rng}.", "",
               "| column | type | meaning | unit | source | stage |", "|---|---|---|---|---|---|"]
        for c, (meaning, unit, source, stage) in columns(name).items():
            md.append(f"| `{c}` | {o['columns'].get(c, '?')} | {meaning} | {unit} | {source} | {stage} |")
        md.append("")
        if stats is not None and name != "column_stats.parquet":
            md += [*column_stats.stats_table(stats, name), "",
                   "Rounded to 4 significant figures; p1 and p99 for factors and multiples. Text columns: unique values "
                   "and the 3 most common with their share of rows (5 in column_stats.parquet).", ""]
        if coverage is not None and name == "factor_panel_monthly.parquet":
            md += [*column_stats.coverage_table(coverage), ""]
    md += sectors()
    return "\n".join(md)


def render_schema() -> str:
    """pipeline/DATA_DICTIONARY.md for the (public) repo: the schema only. Every file and column with meaning, unit,
    source and stage, from this module alone: nothing here comes from the data (no stats, counts or values)."""
    md = ["# Factor-Lake WRDS data: data dictionary (schema)", "",
          "Written by stage `s08_export` from `pipeline/dictionary.py`. It describes every file the pipeline exports "
          "to `processed/final/` and every column in it. It holds no numbers from the data: the full version with "
          "statistics and coverage by year (`processed/DATA_DICTIONARY.md`) stays next to the data, because the data "
          "is licensed from WRDS. All files are Parquet; read them with `pandas.read_parquet`.", "",
          *CONVENTIONS,
          "## Files", "", "| file | one row per | key | built by |", "|---|---|---|---|"]
    for name, f in FILES.items():
        md.append(f"| [{name}](#{_anchor(name)}) | {f['row']} | {', '.join(f['key'])} | `{f['stage']}` |")
    md.append("")
    for name, f in FILES.items():
        md += [f"## {name}", "", f["about"], "",
               f"One row per {f['row']}; key {', '.join(f['key'])}. Built by stage `{f['stage']}`.", "",
               "| column | meaning | unit | source | stage |", "|---|---|---|---|---|"]
        for c, (meaning, unit, source, stage) in columns(name).items():
            md.append(f"| `{c}` | {meaning} | {unit} | {source} | {stage} |")
        md.append("")
    md += sectors()
    return "\n".join(md)


CONVENTIONS = [
          "## Conventions", "",
          "- **Keys:** `permno` is CRSP's security id (one share class, int32); `permco` the company; `gvkey` "
          "Compustat's company id (text). Panels are keyed by a date plus permno.",
          "- **Dates** are `datetime64[ns]` without a time zone. A *month end* is the last CRSP trading day of the "
          "month, not the calendar month end.",
          "- **Units:** returns, yields and most factors are decimals (0.05 = 5%). Money is $ millions unless the "
          "column says per share. Compustat shares are millions, CRSP `shrout` thousands.",
          "- **Point in time:** in factor_panel_monthly every value at date t was public by t (IWM snapshot "
          "effective date, Compustat report date, IBES statistics date, announcement date). The other tables are "
          "not lagged: filter on their report_date / statpers yourself.",
          "- **Which market cap:** `mkt_cap_musd` (the listed security) for portfolio weights; "
          "`company_mkt_cap_musd` (all share classes, unlisted ones from Compustat) for valuation ratios.",
          "- **Missing values** are NaN / null, never 0.", ""]


def sectors() -> list[str]:
    md = ["## Sectors", "", "`gsector` and `gsector_name` are Compustat GICS (the current classification, applied to "
           "all history). Where GICS is missing, `sector_name` maps the CRSP SIC code to a GICS sector name, first "
           "match wins:", "", "| GICS sector | SIC ranges |", "|---|---|",
           "| Unclassified | missing, 0, 9900-9999 |"]
    for name, ranges in SIC_TO_GICS_SECTOR:
        md.append(f"| {name} | {', '.join(f'{lo}-{hi}' if lo != hi else str(lo) for lo, hi in ranges)} |")
    md += ["| Industrials | every other SIC code |", "",
           "GICS sector codes: " + ", ".join(f"{k} {v}" for k, v in GICS_SECTORS.items()) + ".", ""]
    return md


def _anchor(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch in "-_ ").replace(" ", "-")


def parse_columns(md: str) -> dict[str, list[str]]:
    """The columns per file section of a rendered DATA_DICTIONARY.md (the check reads the written file back)."""
    found: dict[str, list[str]] = {}
    section = None
    for line in md.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            section = section if section in FILES else None
        elif line.startswith("### "):                   # Stats and coverage tables come after the column table
            section = None
        elif section and line.startswith("| `"):
            found.setdefault(section, []).append(line.split("`")[1])
    return found
