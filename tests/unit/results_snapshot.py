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

from tests.unit.test_saved_runs import (
    _assert_clean,
    _click_main_tab,
    _open_tab,
    _synthetic_universe,
    browser_like_widget_states,
)

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


# How main_area reads each kind of element, and the value it keeps of each
KINDS = {
    'markdown': (lambda b: b.markdown, lambda e: e.value),
    'caption': (lambda b: b.caption, lambda e: e.value),
    'header': (lambda b: b.header, lambda e: e.value),
    'subheader': (lambda b: b.subheader, lambda e: e.value),
    'info': (lambda b: b.info, lambda e: e.value),
    'success': (lambda b: b.success, lambda e: e.value),
    'warning': (lambda b: b.warning, lambda e: e.value),
    'error': (lambda b: b.error, lambda e: e.value),
    'code': (lambda b: b.code, lambda e: e.value),
    'metric': (lambda b: b.metric, lambda m: [m.label, m.value, m.delta]),
    'dataframe': (lambda b: b.dataframe, lambda d: d.value.to_csv(index=False)),
    'table': (lambda b: b.table, lambda t: t.value.to_csv(index=False)),
    'selectbox': (lambda b: b.selectbox, lambda s: [s.label, [str(o) for o in s.options], str(s.value)]),
    'slider': (lambda b: b.slider, lambda s: [s.label, s.value]),
    'button': (lambda b: b.button, lambda b: b.label),
    'expander': (lambda b: b.expander, lambda e: e.label),
    'tabs': (lambda b: b.tabs, lambda t: t.label),
    'plotly': (lambda b: b.get('plotly_chart'), lambda c: json.loads(c.proto.spec)),
}
# The Results tab's Saved analyses panel: the container whose handle names this checkbox
# (TOGGLE_ID in app/components/saved_panel.py; main's code has no panel)
PANEL_TOGGLE_ID = 'saved-analyses-panel-open'


def _panel_column(at):
    """The Saved analyses panel's container, or None (no runs, or main's code)."""
    def blocks(node):
        for child in getattr(node, 'children', {}).values():
            if hasattr(child, 'children'):
                yield child
                yield from blocks(child)
    return next((b for b in blocks(at.main)
                 if any(getattr(c, 'type', None) == 'html' and PANEL_TOGGLE_ID in str(c.proto)
                        for c in b.children.values())), None)


def main_area(at) -> dict:
    """
    The main area's content, by element type, in document order (the sidebar is left out).
    The Saved analyses panel over the runs is left out too, and given as 'panel', in the
    same form ({} without the panel).
    """
    panel = _panel_column(at)
    in_panel = {id(e) for kind, (elements, _) in KINDS.items() for e in elements(panel)} if panel else set()
    render = {kind: [value(e) for e in elements(at.main) if id(e) not in in_panel]
              for kind, (elements, value) in KINDS.items()}
    render['panel'] = {kind: [value(e) for e in elements(panel)]
                       for kind, (elements, value) in KINDS.items()} if panel else {}
    return render


def moved_to_panel(render: dict, panel: dict, strict: bool = True) -> dict:
    """
    main's rendering without what the Saved analyses panel now shows: main drew those boxes
    above the runs. strict: every item of the panel must be one main showed (it moved, it is
    not new); otherwise items main did not show are ignored (another kind of run's panel).
    """
    def same(value):        # clock times and counts ("This session's analyses (2)") aside
        text = re.sub(r'at \d\d:\d\d:\d\d', 'at <time>', json.dumps(value))
        return re.sub(r' \(\d+\)"$', '"', text)
    render = {kind: list(values) for kind, values in render.items()}
    for kind, values in panel.items():
        for value in values:
            match = next((v for v in render[kind] if same(v) == same(value)), None)
            if match is not None:
                render[kind].remove(match)
            else:
                assert not strict, f"The panel shows a {kind} main did not: {value!r}"
    return render


def without_strategy_controls(render: dict) -> dict:
    """Exclude only the intentional additions, keeping original result snapshots frozen.

    Import settings from file (on the Analysis tab, which the snapshot's steps also hold) and
    each run's Rename and Use these settings have their own AppTest coverage. Every existing
    metric, table, chart and control must still match the main snapshot.
    """
    added = {
        'button': {'Apply', 'Use these settings'},
        'expander': {'Import settings from file'},
        'caption': {
            "Load a settings file (JSON, from a run's Export settings) into these inputs. "
            'It replaces the current inputs and runs nothing.',
            'Names and notes are kept for this session and included in exported settings files.',
        },
    }
    return {kind: [value for value in values if value not in added[kind]] if kind in added else values
            for kind, values in render.items()}


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
