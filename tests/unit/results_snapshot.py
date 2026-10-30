"""
A snapshot of the Results tab for Legacy runs: everything the main area shows (text,
metrics, tables, widgets and Plotly chart specs), in order, for two local Legacy runs
and a cohort comparison on the synthetic universe.

tests/fixtures/legacy_results_render_main.json holds the snapshot taken with main's code,
which has no WRDS mode: first c063100, then c1c3c04 when main gained the analysis lists
(PR #69). PR #67 added its Clear Parameters button to the two steps that show the
Analysis tab (the only change). test_wrds_results.py checks the branch renders the same. This file only uses
helpers that exist on main, so it runs there unchanged (in a worktree of main):

    uv run python -m tests.unit.results_snapshot <output.json>
"""
import csv
import io
import json
import re
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.unit.test_saved_runs import (_assert_clean, _click_main_tab, _open_tab, _synthetic_universe,
                                        browser_like_widget_states)

APP_DIR = Path(__file__).resolve().parents[2] / 'app'


def _cell(text: str):
    """A table cell as a number when it is one, else the text."""
    try:
        return float(text)
    except ValueError:
        return text


def comparable(snapshot: dict) -> dict:
    """
    The snapshot in a form that compares the same on every machine: the session history's
    clock times ("at 14:03:59") masked, and each table's CSV parsed into rows of cells
    (pandas writes CSV lines ending in \\r\\n on Windows and \\n on Linux, and a number can
    differ in its last binary digit between them). Compare with a numeric tolerance.
    """
    snapshot = json.loads(re.sub(r'at \d\d:\d\d:\d\d', 'at <time>', json.dumps(snapshot)))
    for step in snapshot.values():
        for kind in ('dataframe', 'table'):
            step[kind] = [[[_cell(c) for c in row] for row in csv.reader(io.StringIO(text))]
                          for text in step[kind]]
    return snapshot


def _text(elements):
    return [e.value for e in elements]


def main_area(at) -> dict:
    """The main area's content, by element type, in document order (the sidebar is left out)."""
    main = at.main
    return {
        'markdown': _text(main.markdown),
        'caption': _text(main.caption),
        'header': _text(main.header),
        'subheader': _text(main.subheader),
        'info': _text(main.info),
        'success': _text(main.success),
        'warning': _text(main.warning),
        'error': _text(main.error),
        'code': _text(main.code),
        'metric': [[m.label, m.value, m.delta] for m in main.metric],
        'dataframe': [d.value.to_csv(index=False) for d in main.dataframe],
        'table': [t.value.to_csv(index=False) for t in main.table],
        'selectbox': [[s.label, [str(o) for o in s.options], str(s.value)] for s in main.selectbox],
        'slider': [[s.label, s.value] for s in main.slider],
        'button': [b.label for b in main.button],
        'expander': [e.label for e in main.expander],
        'tabs': [t.label for t in main.tabs],
        'plotly': [json.loads(c.proto.spec) for c in main.get('plotly_chart')],
    }


def legacy_results_snapshot() -> dict:
    """Runs two Legacy analyses locally and a cohort comparison; returns the Results tab after each step."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', '')
        mp.syspath_prepend(str(APP_DIR))
        browser_like_widget_states(mp)
        at = AppTest.from_file(str(APP_DIR / 'streamlit_app.py'), default_timeout=120)
        at.session_state['raw_data'] = _synthetic_universe()
        at.run()
        snapshot = {}

        # Run 1: Mixing, equal weight, two factors
        for key in ('roe', 'roa_pct'):
            at.button(key=f'factor_add_{key}').click().run()
        next(b for b in at.button if b.label == 'Load Market Data').click().run()
        next(b for b in at.button if b.label == 'Run Portfolio Analysis').click().run()
        _assert_clean(at)
        snapshot['run1'] = main_area(at)

        # Run 2: Integrating, market cap weight, Hold Cash
        _click_main_tab(at, 'Analysis')
        next(r for r in at.sidebar.radio if r.key == 'factor_combination_input').set_value('Integrating').run()
        next(r for r in at.sidebar.radio if r.label == 'Select weighting method:').set_value('Market Cap Weight').run()
        next(r for r in at.sidebar.radio if r.label == 'How to treat delisted positions:').set_value('Hold Cash').run()
        _click_main_tab(at, 'Analysis')
        next(b for b in at.button if b.label == 'Load Market Data').click().run()
        next(b for b in at.button if b.label == 'Run Portfolio Analysis').click().run()
        _assert_clean(at)
        snapshot['run2'] = main_area(at)

        # Run 2's cohort comparison, then back to run 1 with another year
        run2 = at.session_state['saved_runs'][0]
        at.button(key=f"cohort_btn_{run2['id']}").click().run()
        _assert_clean(at)
        snapshot['run2_cohort'] = main_area(at)
        run1 = at.session_state['saved_runs'][1]
        _open_tab(at, run1['label'])
        at.selectbox(key=f"ranked_year_{run1['id']}").set_value(2010).run()
        _assert_clean(at)
        snapshot['run1_2010'] = main_area(at)
        return snapshot


if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps(legacy_results_snapshot(), indent=1, sort_keys=True) + '\n',
                                 encoding='utf-8')
    print(f"Wrote {sys.argv[1]}")
