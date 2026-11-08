# WRDS pulls for data version 2 (wrds_2026_11)

How every raw file in `raw_v2/` was pulled (2026-11-07), so the pulls can be rerun or extended. Each WRDS query
is saved in the Query Manager:
`https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/<ID>/` (Cornell proxy; log in
first). The only numbers here are row counts, to check a rerun against. The decisions behind these dates are 33-41
in [PLAN.md](PLAN.md); data version 1 (wrds_2026_10) used the same products with shorter date ranges (its query list
is in `pipeline/README.md` at the tag `data-wrds_2026_10`). The exact columns the pipeline expects are `RAW_FILES` in
`pipeline/config.py`: s00_inventory FAILs on a missing column, so a changed query shows up before any cleaning runs.

## Upload lists

| list | lines | how it was built |
|---|---|---|
| `raw/query_inputs/permno_list_v2.txt` | 7,157 | the v1 list (every IWM snapshot permno), plus every permno of the client's 9/30 lists (s00c_client_map), plus the IWM known-gap fixes (s01 rule 2b) |
| `raw/query_inputs/ibes_tickers_v2.txt` | 6,978 | from the new IBES link pull (11882519): every IBES ticker linked to a v2 permno whose link ends on or after 2001-01-01 or has no end date, any score |

Order: pull the IBES link first (with permno_list_v2), build ibes_tickers_v2 from it, then pull IBES. Every other
pull can run at the same time. The names history is used to map the client lists, so it comes before
permno_list_v2.

## The queries

| file in raw_v2/ | query ID | WRDS product | rows |
|---|---|---|---|
| r2000_price_crsp_ciz.csv | [11882814](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882814/) | crsp_m_stock | 23,660,957 |
| delistings.csv | [11882730](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882730/) | crsp_m_stock | 4,352 |
| benchmark_etfs.csv | [11882737](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882737/) | crsp_m_stock | 19,800 |
| forward_estimates.csv | [11882787](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882787/) | tr_ibes | 11,551,427 |
| crsp_ibes_link.csv | [11882519](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882519/) | wrdsapps_link_crsp_ibes | 12,016 |
| fundamentals_raw.csv | [11882617](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882617/) | crsp_a_ccm | 349,389 |
| fundamentals_annual_raw.csv | [11882638](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882638/) | crsp_a_ccm | 85,959 |
| index_close.csv | [11882694](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882694/) | comp_na_daily_all | 22,551 |
| ff_daily.csv | [11882698](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882698/) | ff_all | 7,483 |
| ff_monthly.csv | [11882700](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882700/) | ff_all | 357 |
| crsp_names_history.csv | [11882425](https://wrds-www-wharton-upenn-edu.proxy.library.cornell.edu/query-manager/query/11882425/) | crsp_a_stock | 191,048 |
| fred_DGS10.csv, fred_DGS3MO.csv, fred_DBAA.csv, fred_DAAA.csv | FRED, not WRDS | - | 7,786 each |
| iwm_holdings_raw.csv | not re-pulled (the 2026-10 pull; see 12) | crsp_q_mutualfunds | 432,497 |

### 1. r2000_price_crsp_ciz.csv (11882814)

- **Form:** CRSP > Monthly Update > Stock - Version 2 > Daily Stock File. Compressed output.
- **Codes:** permno_list_v2.txt (PERMNO).
- **Dates:** 1997-01-01 to the latest (the file starts 1997-01-02; the target was 1997-09-01, for 60-month betas
  from 2002).
- **Variables:** permno, HdrCUSIP, PrimaryExch, SecurityNm, DelReasonType, Ticker, PERMCO, SICCD, DlyCalDt,
  DlyDelFlg, DlyPrc, DlyRet, DlyVol, DlyClose, DlyLow, DlyHigh, DlyOpen, ShrOut, DlyCumFacPr, DlyCumFacShr. The
  query also returned ewretd, ewretx and sprtrn (market returns), which the pipeline does not use: untick them
  next time.
- Use the Monthly Update, not the Annual Update: the annual version ends at the last 12/31 and restates history
  (cumulative factors, shares) for that version.

### 2. delistings.csv (11882730)

- **Form:** CRSP > Monthly Update > Stock - Version 2 > Delisting Information.
- **Codes:** permno_list_v2.txt (permno). **Dates:** all.
- **Variables:** delactiontype, permno, delistingdt, delstatustype, delreasontype, delpaymenttype, delpermno,
  delret, delretmisstype (delstatustype, delpermno and delretmisstype are not used).

### 3. benchmark_etfs.csv (11882737)

- **Form:** CRSP > Monthly Update > Stock - Version 2 > Daily Stock File.
- **Codes:** PERMNO 88222 (IWM), 88399 (IWN), 88401 (IWO).
- **Dates:** 2000-05-01 to the latest (IWM lists in 2000-05, IWN and IWO in 2000-07).
- **Variables:** PERMNO, HdrCUSIP, Ticker, PERMCO, DlyCalDt, DlyPrc, DlyRet.
- The file has a few exact duplicate rows; s06 drops them.

### 4. forward_estimates.csv (11882787)

- **Form:** IBES > IBES Academic > Summary History > Summary Statistics, adjusted, US file.
- **Codes:** ibes_tickers_v2.txt (TICKER).
- **Dates:** statpers 2001-01-01 to the latest.
- **Screens:** measures EPS, CPS, CPX, DPS, EBI, EBT, SAL; FPI 0, 1, 2, 3.
- **Variables:** ticker, CUSIP, OFTIC, STATPERS, MEASURE, FISCALP, FPI, CURCODE, NUMEST, NUMUP, NUMDOWN, MEDEST,
  MEANEST, STDEV, HIGHEST, LOWEST, FPEDATS, ACTUAL, ANNDATS_ACT.
- Expected thin data, not errors (decision 41): CPX starts in 2006; EBI, DPS and CPS are thin in 2001-2004.

### 5. crsp_ibes_link.csv (11882519)

- **Form:** Linking Suite by WRDS > IBES CRSP Link.
- **Codes:** permno_list_v2.txt (PERMNO). **Dates:** all.
- **Variables:** TICKER, PERMNO, ncusip, sdate, edate, SCORE.
- Links ending on the table's last update (2025-12-31) are treated as open-ended (decision 40).

### 6. fundamentals_raw.csv (11882617)

- **Form:** CRSP > Annual Update > CRSP/Compustat Merged > Fundamentals Quarterly.
- **Codes:** permno_list_v2.txt (LPERMNO).
- **Dates:** datadate 1999-01-01 to the latest.
- **Screens:** consolidation C, industry format INDL, data format STD, population source D; currency USD and CAD;
  active and inactive companies; link types LC and LU; fiscal period end within the link range.
- **Variables:** GVKEY, LPERMNO, datadate, fyearq, fqtr, indfmt, consol, popsrc, datafmt, tic, conm, ajexq,
  curcdq, datacqtr, datafqtr, rdq, actq, apq, atq, ceqq, cheq, cogsq, cshfdq, cshoq, dlcq, dlttq, dpq, epsfxq,
  epspxq, gdwlq, ibq, intanq, invtq, lctq, ltq, mibtq, niq, oiadpq, oibdpq, piq, ppentq, pstkq, rectq, revtq,
  saleq, seqq, txditcq, txtq, xintq, xrdq, xsgaq, aqcy, capxy, dltisy, dltry, dvy, oancfy, prstkcy, sstky, costat,
  dvpsxq, prccq, prchq, prclq, ggroup, gind, gsector, gsubind, naics (prchq and prclq are not used).

### 7. fundamentals_annual_raw.csv (11882638)

- **Form:** CRSP > Annual Update > CRSP/Compustat Merged > Fundamentals Annual.
- **Codes, dates and screens:** as the quarterly query.
- **Variables:** GVKEY, LPERMNO, datadate, fyear, indfmt, consol, popsrc, datafmt, tic, conm, ajex, curcd, fyr,
  act, ap, aqc, at, capx, ceq, che, cogs, cshfd, csho, dlc, dltis, dltr, dltt, dp, dv, dvc, emp, epsfx, epspx,
  gdwl, ib, intan, invt, lct, lt, mib, ni, oancf, oiadp, oibdp, pi, ppent, prstkc, pstk, rect, revt, sale, seq,
  sstk, txditc, txt, xint, xintd, xrd, xsga, costat, sich, dvpsx_f, prcc_f, ggroup, gind, gsector, gsubind, naics.

### 8. index_close.csv (11882694)

- **Form:** Compustat > North America > Index Prices - Daily.
- **Codes:** tic I0124 (Russell 2000), I0120 (Russell 2000 Growth), I0122 (Russell 2000 Value).
- **Dates:** 1997-01-01 to the latest. **Variables:** tic, datadate, gvkeyx, prccd.

### 9-10. ff_daily.csv (11882698) and ff_monthly.csv (11882700)

- **Form:** Fama-French Portfolios and Factors > Fama-French 5 Factors Plus Momentum, daily and monthly.
- **Dates:** 1997-01-01 to the latest.
- **Variables:** date (daily) or dateff (monthly), mktrf, smb, hml, rmw, cma, rf, umd.

### 11. crsp_names_history.csv (11882425)

- **Form:** CRSP > Annual Update > Stock - Version 2 (CIZ) > Names (security information history).
- **Codes:** all permnos. **Dates:** all.
- **Variables:** hdrcusip9, permno, permco, secinfostartdt, secinfoenddt, cusip, ticker, issuernm, shareclass,
  issuertype, securitytype, sharetype, primaryexch, tradingstatusflg.
- Used to map the client lists to permnos (s00c) and to decide who is trading on a date (s01). The Annual Update
  ends at 2025-12-31, which is enough for the lists (2002-2023).

### FRED (not WRDS)

`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>&cosd=1997-01-01` for DGS10, DGS3MO, DBAA and DAAA,
saved as `fred_<SERIES>.csv` (columns observation_date and the series, in percent).

### 12. iwm_holdings_raw.csv (not re-pulled for version 2)

- **Form:** CRSP > Mutual Funds > Portfolio Holdings (crsp_q_mutualfunds).
- **Codes:** crsp_portno 1004185 and 1025818 (IWM's two fund ids).
- **Dates:** report dates 2008-02 to the latest.
- **Variables:** crsp_portno, report_dt, eff_dt, percent_tna, nbr_shares, security_name, cusip, permno, ticker.
- Version 2 kept the 2026-10 pull (its query ID was not recorded): membership before 2008-02 comes from the client
  lists, so the holdings did not need a longer range. A refresh pulls it first, to the latest report date, because
  the permno list is built from it ([DATA_UPDATE_GUIDE.md](DATA_UPDATE_GUIDE.md), step 1).

### The client lists (not WRDS)

`raw/client/FR2000 Annual Quant Data(Data).csv`: the client's Russell 2000 lists on each 9/30, from FactSet.
Licensed: never commit it or anything derived from it.

## Superseded queries (do not rerun)

| query ID | what it was |
|---|---|
| 11882501 | prices from the Annual Update (ends 2025-12-31) |
| 11882512 | delistings from the Annual Update (ends 2025-12-31) |
| 11882685 | a wrong "ETF" pull (it was the stock query) |
| 11882728 | prices without DlyOpen |
| 11882783 | prices, the same as the final query but uncompressed |
| 11882746 | IBES without HIGHEST and LOWEST |
| 11882540 | an unused test pull |
| 11882591, 11882813 | ended with errors |

## The GICS gap

No historical GICS is available to us (Compustat's historical GICS table is not in the subscription), so every
row carries the company's current, or last, GICS classification from the Compustat header (gics_source =
current_header, decision 39). A company that changed sector is shown in its latest sector for all of history.
If historical GICS becomes available, pull it by gvkey (the gvkeys of the fundamentals files) and replace the
header codes by the codes valid at each date.
