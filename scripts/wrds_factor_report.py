"""Build the WRDS crosswalk/default report, optionally with private empirical evidence.

uv run python scripts/wrds_factor_report.py --catalog-only --output DOCS/WRDS_FACTOR_REVIEW.md
uv run python scripts/wrds_factor_report.py --s3 --output ../private_reports/wrds_factor_review.md
uv run python scripts/wrds_factor_report.py --data-dir /path/to/processed/final --output /private/report.md

Real-data output must be outside the public repository (pipeline/README.md).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_config import WRDS_FACTOR_METADATA  # noqa: E402
from app.wrds_factor_review import factor_rows, legacy_rows  # noqa: E402
from src.wrds_data import PANEL_FILE, RETURNS_FILE, wrds_dir  # noqa: E402
from src.wrds_factor_evidence import monthly_spreads  # noqa: E402
from src.wrds_factors import WRDS_FACTOR_COLUMNS  # noqa: E402

PANEL_COLUMNS = tuple(dict.fromkeys(('date', 'permno', 'prc', 'mkt_cap_musd') + WRDS_FACTOR_COLUMNS))
RETURNS_COLUMNS = ('month_end', 'permno', 'ret_m')

METHODOLOGY = '''Monthly equal-weight tails, 10% by default, with at least one stock per leg and no overlap.
Sort finite raw factor values at month end t; use returns in the next calendar month (t+1), including
CRSP delisting returns. Break ties by PERMNO. Use the existing buyability filter (positive price and market cap),
without a sector or fossil-fuel restriction. No forward-return column is used in selection.
For Higher defaults: long high, short low. For Lower defaults: long low, short high.
Reported spread is 12 times the arithmetic mean of monthly long-minus-short returns, in percentage points
per year; it is NOT a difference of CAGRs and is NOT a compounded trading strategy.
Each leg has $1 notional (gross exposure $2); borrowing fees, financing, transaction costs and taxes are excluded.
Missing individual returns retain their weight and are filled with 0% as in the app, with a count and coverage flag.
Months with fewer than two finite signals, a constant signal, or no next-month return-file coverage are skipped
and counted. Actual factor coverage is shown; partial 2002 and 2026 years are not presented as full years.
This diagnostic uses raw single-factor ranks, not the app's winsorized multi-factor scoring.
Negative default-oriented spreads are marked for team review; no default is flipped automatically.
Positive spreads are descriptive evidence, not proof of significance or out-of-sample performance.
'''
REFERENCES = '''- Internal definitions: `app/streamlit_config.py`, `src/factor_registry.py`,
  `pipeline/DATA_DICTIONARY.md`, and `pipeline/stages/s02_prices.py`.
- Existing default decisions: `DOCS/WRDS_MODE_DESIGN.md`, decision 4 and section 2.1.
- Kenneth French, [Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):
  value, size, profitability, investment, momentum and reversal constructions. These are conceptual references,
  not claims that this report replicates the published factor portfolios.
- French, [Momentum](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_mom_factor.html).
- Asness, Frazzini and Pedersen, [Quality Minus Junk](https://www.aqr.com/insights/research/working-paper/quality-minus-junk).
- Frazzini and Pedersen, [Betting Against Beta](https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta).

Rationales retain the project's existing theory-based choices. Liquidity, sales growth, R&D intensity,
analyst disagreement and long-term forecasts are explicitly context-dependent conventions; none is asserted
to be universally optimal. The evidence is for team review, not automatic sign selection.
'''


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(_cell(value) for value in row) + ' |' for row in rows])


def render_report(summaries=None, *, source='No data loaded', start_year=2002, end_year=2026, tail_pct=10):
    lines = ['# WRDS factor matching and default directions', '',
             '## Review status', '',
             ('Empirical evidence computed from the source identified below. Team review is still required for '
              'disagreements and coverage flags.' if summaries is not None else
              '**Empirical spreads pending: this catalog contains no real-data results.** '
              'Run the accompanying script against the authorized WRDS data to complete the evidence.'), '',
             'Matches are based on documented formula and lookback, not equal vendor values. Units, '
             'publication timing and coverage can differ. Similar names, reciprocals and undocumented '
             'lookbacks are related candidates only. Unmatched factors are listed, not rebuilt.', '',
             f'Requested return window: **{start_year}–{end_year}**. Tail size: **{tail_pct:g}%**.',
             f'Source: {_cell(source)}', '', '## WRDS factors', '']
    rows = []
    for row in factor_rows():
        evidence = (summaries or {}).get(row['column'])
        spread = 'pending'
        coverage = 'not computed'
        status = 'pending data'
        if evidence is not None:
            value = evidence['annualized_spread']
            spread = 'n/a' if value is None else f'{100 * value:+.3f} pp/year'
            coverage = (f"{evidence['first_return_month'] or 'n/a'} to {evidence['last_return_month'] or 'n/a'}; "
                        f"{evidence['months']} months; {evidence['skipped_months']} skipped; "
                        f"{evidence['missing_returns']} missing stock returns")
            status = f"{evidence['evidence_status']}; {evidence['coverage_status']}"
        rows.append([row['factor'], f"`{row['column']}`", row['legacy_match'], row['default_direction'],
                     row['reason'], spread, coverage, status])
    lines += [_table(['WRDS factor', 'Column', 'Legacy match', 'Default', 'Reason', 'Long–short spread',
                      'Coverage', 'Review'], rows), '', '## Every Legacy factor', '',
              _table(['Legacy factor', 'Confirmed WRDS match', 'Related candidate only', 'Definition / limitation'],
                     [[r['legacy_factor'], r['wrds_match'], r['related_candidate'] or '—', r['note']]
                      for r in legacy_rows()]), '', '## Evidence methodology', '', METHODOLOGY,
              '## Sources and interpretation', '', REFERENCES,
              '## Generate the private evidence report', '',
              'Run from the repository root, using your existing local AWS configuration:', '',
              '```bash', 'uv run python scripts/wrds_factor_report.py --s3 '
              '--output ../private_reports/wrds_factor_review.md', '```', '',
              'Or replace `--s3` with `--data-dir "/path/to/processed/final"`. The script also writes a '
              'monthly CSV and a JSON provenance/summary file beside the report. '
              'Keep all real-data outputs outside this public repository. No defaults or data files are modified.', '']
    return '\n'.join(lines)


def _load_aws_settings():
    # Standard AWS environment/profile wins. Also support the app's existing local secrets file.
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    path = PROJECT_ROOT / '.streamlit/secrets.toml'
    if path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with path.open('rb') as handle:
            config = tomllib.load(handle)
        for key in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_DEFAULT_REGION',
                    'AWS_PROFILE', 'FACTOR_LAKE_S3_BUCKET', 'FACTOR_LAKE_S3_PREFIX', 'FACTOR_LAKE_WRDS_DIR'):
            if key in config:
                os.environ.setdefault(key, str(config[key]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--catalog-only', action='store_true')
    mode.add_argument('--data-dir', type=Path)
    mode.add_argument('--s3', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-year', type=int, default=2002)
    parser.add_argument('--end-year', type=int, default=2026)
    parser.add_argument('--tail-pct', type=float, default=10.0)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not args.catalog_only and (output == PROJECT_ROOT or PROJECT_ROOT in output.parents):
        parser.error('Real-data reports must be outside the public repository.')
    if output.suffix.lower() != '.md':
        parser.error('--output must end in .md')
    summaries, monthly, provenance = None, [], {}
    source = 'No data loaded (catalog only)'
    if not args.catalog_only:
        if args.s3:
            _load_aws_settings()
            from src.s3_client import S3DataManager
            manager = S3DataManager()
            folder = wrds_dir()
            def read_bytes(name):
                return manager.read_bytes(f'{folder}/{name}')
            source = manager.s3_uri(folder)
        else:
            folder = args.data_dir.resolve()
            def read_bytes(name):
                return (folder / name).read_bytes()
            source = str(folder)
        frames = []
        for name, columns in ((PANEL_FILE, PANEL_COLUMNS), (RETURNS_FILE, RETURNS_COLUMNS)):
            raw = read_bytes(name)
            provenance[name] = {'sha256': hashlib.sha256(raw).hexdigest()}
            frames.append(pd.read_parquet(io.BytesIO(raw), columns=list(columns)))
            del raw
        directions = {m['column']: m['higher_is_better'] for m in WRDS_FACTOR_METADATA.values()}
        summaries, monthly = monthly_spreads(*frames, directions, start_year=args.start_year,
                                            end_year=args.end_year, tail_pct=args.tail_pct)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(summaries, source=source, start_year=args.start_year,
                                    end_year=args.end_year, tail_pct=args.tail_pct), encoding='utf-8')
    if summaries is not None:
        payload = {'generated_at': datetime.now(timezone.utc).isoformat(), 'source': source,
                   'inputs': provenance, 'requested_years': [args.start_year, args.end_year],
                   'tail_pct': args.tail_pct, 'methodology': METHODOLOGY, 'factors': summaries}
        output.with_suffix('.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        fields = ['column', 'signal_date', 'return_month', 'eligible_signals', 'stocks_per_leg',
                  'high_return', 'low_return', 'long_return', 'short_return', 'spread', 'missing_returns']
        with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(monthly)
    print('Report written. Defaults and source data were not modified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
