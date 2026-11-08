# GICS mapping (WRDS mode)

How the WRDS panel's GICS codes and names are built, and how the app's GICS filter uses them. For the person who runs
a data update or changes the filter. The user-facing description is in the
[user guide](FACTOR_LAKE_USER_GUIDE.md#filter-by-gics-industry).

## Where the codes come from

The panel's GICS codes (`gsector`, `ggroup`, `gind`, `gsubind`) come from the Compustat quarterly fundamentals
(`s03`, used at `s07`): the fundamentals row used at month end t, else the stock's latest row with GICS. Compustat
only carries each company's **current** classification (the last one, for an inactive company), so that one code is
applied to the company's whole history: every row with GICS has `gics_source = current_header` (PLAN.md decisions 4
and 39). **There is no historical GICS.** A company that changed industry is filtered by its latest industry in every
month, so the GICS filter is the one WRDS input that is not strictly point in time (WRDS_MODE_DESIGN.md, risk R3).
The app says this once, in the Filter level help text.

## How the four levels relate

The levels nest by digits. The 8-digit sub-industry code holds the others: its first 2, 4 and 6 digits are its
sector, industry group and industry.

| Level | Column | Digits | Example |
|---|---|---|---|
| Sector | `gsector` | 2 | 40 Financials |
| Industry group | `ggroup` | 4 | 4010 Banks |
| Industry | `gind` | 6 | 401010 Banks |
| Sub-industry | `gsubind` | 8 | 40101015 Regional Banks |

The app works out a parent from the code alone (`parent_code` in `src/wrds_gics.py`), so the sidebar's lists narrow
by prefix and a filter only needs `gsubind`.

## How names are attached

- **Pipeline:** the code-to-name tables `GICS_SECTORS`, `GICS_GROUPS`, `GICS_INDUSTRIES` and `GICS_SUB_INDUSTRIES` in
  [`pipeline/config.py`](https://github.com/cornell-sysen-5900/Factor-Lake/blob/main/pipeline/config.py). `add_gics_names` in
  [`pipeline/stages/s03_fundamentals_q.py`](https://github.com/cornell-sysen-5900/Factor-Lake/blob/main/pipeline/stages/s03_fundamentals_q.py)
  maps them onto the `*_name` columns (s04 and s07 call it too). Names follow the 2023 GICS structure. A retired code
  keeps its last name with the year it was dropped, e.g. "Media (discontinued 2018)".
- **App:** [`src/wrds_gics.py`](https://github.com/cornell-sysen-5900/Factor-Lake/blob/main/src/wrds_gics.py) holds a
  copy of the four tables, limited to the codes in the panel. The sidebar lists, the job validation and the
  Universe / Compare labels use it. The Lambda does not import `pipeline/`, so the app needs its own copy.
  `test_tables_match_the_pipelines_names_and_nest` in `tests/unit/test_wrds_gics.py` checks that every name equals the
  pipeline's, that names are unique within a level (the lists show names) and that every code's parents exist.
- **Holdings** show the panel's own `*_name` columns, so they always carry the pipeline's names.

## Filter rules

A WRDS job's `gics_filter` is `{"level": ..., "codes": [...]}`, and saved filters store codes, not names.

- **Sector** keeps the stocks whose `sector_name` is one of the chosen sectors. `sector_name` is the GICS sector,
  or, where a stock has no GICS (`gics_from_sic = True`), its CRSP SIC code mapped to a sector by
  `SIC_TO_GICS_SECTOR` in `pipeline/config.py` (decision 4). This is exactly the older `selected_sectors` filter, so
  old jobs and files give the same result. Only Unclassified stocks (no GICS and no usable SIC code) are left out.
- **Industry group, industry and sub-industry** keep the stocks whose `gsubind` starts with one of the chosen codes.
  Stocks without GICS are left out, because a SIC code has no finer GICS equivalent.

How much this affects, in `wrds_2026_11`: about **0.09%** of panel rows have no GICS, so the finer levels leave out
fewer than 1 row in 1,000 that the Sector level keeps. About **0.01%** of rows are Unclassified and are left out
whenever any GICS filter is on. Every row with a sector code also has a sub-industry code. The by-year shares
are in section 11 of the version's private `VALIDATION.md`.

## At each data update

Section 11 (GICS) of `processed_vN/s09_validation/VALIDATION.md` and the s03 summary list what changed. Check:

1. **Codes with no name.** s03 WARNs "every GICS gsubind code has a name" (one check per level, listing the codes)
   and s09 "GICS: every code of the four levels has a name". Add each code to its table in `pipeline/config.py`
   (the official MSCI name, unique within its level) and rerun from s03. Until then the stock keeps its code, but
   its name is blank in the holdings and the code can't be picked in the sidebar.
2. **New codes.** Compare the panel's distinct codes at each level with the tables in `src/wrds_gics.py` and add the
   missing ones with the pipeline's names. Until then those stocks pass a Sector filter, and pass a finer filter
   only through a parent level that is listed. A strategy file that names an unknown code is rejected.
3. **Retired or renumbered codes.** Compustat keeps an inactive company's last code, so retired codes stay in the
   panel. Keep them in both tables, with the name marked "(discontinued YYYY)". Never delete a code from
   `src/wrds_gics.py`: strategy files and reruns that use it would be rejected. A company that MSCI
   renumbered moves to its new code for its whole history, so a saved filter on the old code may now hold fewer
   stocks. Say so in the release note.
4. **Renamed industries.** Change the name in `pipeline/config.py` and in `src/wrds_gics.py` together (the test
   above fails otherwise). Filters are stored as codes, so saved analyses and strategy files are not affected.
5. **Coverage.** The share of rows with each level and the share from the SIC fallback should be close to the
   previous version's. A jump means a Compustat pull or link problem, not a change in GICS.

Then run `uv run pytest tests/unit/test_wrds_gics.py` and the sidebar tests
(`tests/unit/test_wrds_gics_sidebar.py`).
