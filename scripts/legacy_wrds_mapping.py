"""Generate the definition-match report from the existing reviewed factor catalog."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.streamlit_config import FACTOR_METADATA, WRDS_FACTOR_METADATA  # noqa: E402
from app.wrds_factor_review import REPLACEMENTS, SAME  # noqa: E402


def report():
    names = {m['column']: name for name, m in WRDS_FACTOR_METADATA.items()}
    replacements = {r.legacy_column: r for r in REPLACEMENTS}
    lines = ['# Legacy-to-WRDS definition matches', '',
             'New runs use WRDS. Saved Legacy analyses remain viewable by ID in Results and Compare.', '',
             'This report distinguishes a documented **Same** definition from a **Similar** alternative in the',
             'existing reviewed catalog. `none` means there is no confirmed definition match; a related factor',
             'may still be available. Similar factors are not rebuilt or silently substituted. Units, dates,',
             'vendor values and coverage can differ even for a Same match.', '',
             '| Legacy factor | WRDS definition match | Related WRDS factor | Notes |',
             '|---|---|---|---|']
    for name, meta in FACTOR_METADATA.items():
        r = replacements.get(meta['column'])
        match = names[r.wrds_column] if r and r.level == SAME else 'none'
        related = names[r.wrds_column] if r and r.level != SAME else '—'
        note = r.difference if r else 'No reviewed mapping; listed for team review, not rebuilt.'
        lines.append(f'| {name} | {match} | {related} | {note} |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    print(report(), end='')
