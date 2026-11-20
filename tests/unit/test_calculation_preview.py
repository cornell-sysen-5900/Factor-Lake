"""Acceptance tests for live strategy formulas and historical run snapshots."""
import copy
import re

import pytest

import app.streamlit_utils as utils
import src.backtest_engine as engine
import src.backtest_jobs as jobs
import src.wrds_service as service
from app.calculation_preview import (
    PREVIEW_LABEL,
    SAVED_LABEL,
    STEPS_LABEL,
    build_calculation_preview,
    preview_for_run,
)
from app.streamlit_config import FACTOR_METADATA
from src.backtest_jobs import run_job
from src.wrds_calendar import FREQUENCY_LABELS
from tests.unit.test_local_run_filters import state  # noqa: F401 (fixture)
from tests.unit.test_saved_runs import (
    _add_factor,
    _assert_clean,
    _click,
    _click_main_tab,
    at,
)
from tests.unit.test_wrds_results import s3  # noqa: F401 (fixture)
from tests.unit.test_wrds_saved import manager, new_session  # noqa: F401 (fixtures)


def formula(at):
    return next(c.value for c in at.code if "U = market data" in c.value)


def control(elements, label):
    return next(e for e in elements if e.label == label)


def outline(at):
    """The main area's headers, subheaders, expanders and buttons, in document order."""
    out = []
    for node in at.main:
        if type(node).__name__ == "Expander" or node.type == "button":
            out.append(node.label)
        elif node.type in ("header", "subheader"):
            out.append(node.value)
    return out


def preview_box(at, label):
    """The preview's expander: collapsed, with the formula and the steps in a nested, collapsed expander."""
    box = next(e for e in at.main.expander if e.label == label)
    assert box.proto.expanded is False
    [steps] = [e for e in box.expander if e.label == STEPS_LABEL]
    assert steps.proto.expanded is False
    assert len(box.code) == 1 and any(m.value.startswith("**1. ") for m in steps.markdown)
    return box


def test_preview_is_one_collapsed_expander_that_does_not_compete(at):
    # Analysis: below Factor Selection, collapsed, with Load Market Data and Run right after it
    _add_factor(at, "ROA %")
    preview_box(at, PREVIEW_LABEL)
    shown = outline(at)
    assert shown.index("Factor Selection") < shown.index(PREVIEW_LABEL) < shown.index("Load Market Data")
    # Results: the results first, the run's calculation collapsed below Performance Summary
    _click(at, "Load Market Data")
    _click(at, "Run Portfolio Analysis")
    _assert_clean(at)
    box = preview_box(at, SAVED_LABEL)
    assert box.code[0].value == at.session_state["saved_runs"][0]["calculation_preview"]["formula"]
    shown = outline(at)
    assert shown.index("Performance Summary") < shown.index(SAVED_LABEL) < shown.index("Holdings and Ranks")


def test_preview_before_loading_and_live_factor_terms(at):
    assert "[select at least one factor]" in formula(at)
    assert not at.session_state["data_loaded"]
    assert "Step-by-step calculation" in [e.label for e in at.expander]
    _add_factor(at, "ROA %")
    _add_factor(at, "ROE using 9/30 Data")
    assert "Sleeve[ROA %]" in formula(at)
    assert "Sleeve[ROE using 9/30 Data]" in formula(at)
    key = FACTOR_METADATA["ROA %"]["key"]
    at.button_group(key=f"factor_dir_{key}").set_value("Lower is better").run()
    assert "-Winsorize_1%(ROA %" in formula(at)
    at.button(key=f"factor_remove_{key}").click().run()
    assert "ROA %" not in formula(at)
    assert "ROE using 9/30 Data" in formula(at)
    assert not at.session_state["data_loaded"]
    _assert_clean(at)


def test_sidebar_changes_update_formula_without_data(at):
    _add_factor(at, "ROA %")
    control(at.radio, "Select weighting method:").set_value("Market Cap Weight").run()
    assert "w_i = MarketCap_i /" in formula(at)
    control(at.radio, "Combine factors").set_value("Integrating").run()
    assert "Integrating: Score(i,t) = (z1) / 1" in formula(at)
    for label, expected in [("Hold Cash", "risk-free rate for year t"),
                            ("Reinvest", "value-weighted return of surviving"),
                            ("Zero Return", "post-delist rate q = 0")]:
        control(at.radio, "How to treat delisted positions:").set_value(label).run()
        assert expected in formula(at)
    control(at.checkbox, "Restrict Fossil Fuel Companies").check().run()
    assert "ESG = exclude fossil fuel industries" in formula(at)
    control(at.checkbox, "Enable Sector Filter").check().run()
    control(at.multiselect, "Include following sectors:").set_value(["Technology"]).run()
    assert "sectors = Technology;" in formula(at)
    control(at.multiselect, "Include following sectors:").set_value([]).run()
    assert "all sectors (no sector restriction)" in formula(at)
    at.number_input(key="start_year_input").set_value(2010).run()
    assert "Year in [2010, 2024]" in formula(at)
    control(at.number_input, "Initial AUM ($)").set_value(2500.0).run()
    assert "V(2010) = $2,500.00" in formula(at)
    assert not at.session_state["data_loaded"]
    _assert_clean(at)


def test_results_keep_exact_formula_after_sidebar_changes(at):
    _add_factor(at, "ROA %")
    _click(at, "Load Market Data")
    # Change filters after loading: actual run and preview must still agree.
    control(at.checkbox, "Enable Sector Filter").check().run()
    control(at.multiselect, "Include following sectors:").set_value(["Technology"]).run()
    at.number_input(key="start_year_input").set_value(2010).run()
    expected = formula(at)
    _click(at, "Run Portfolio Analysis")
    _assert_clean(at)
    assert formula(at) == expected
    run = at.session_state["saved_runs"][0]
    assert run["data"]["Year"].min() == 2010
    assert run["universe"]["selected_sectors"] == ["Technology"]
    assert run["calculation_preview"]["formula"] == expected
    control(at.radio, "Select weighting method:").set_value("Market Cap Weight").run()
    assert formula(at) == expected
    _click_main_tab(at, "Analysis")
    assert formula(at) != expected
    _click_main_tab(at, "Results")
    assert formula(at) == expected
    _assert_clean(at)


@pytest.mark.parametrize("mode", ["mixing", "integrating"])
def test_older_and_opened_runs_use_stored_inputs(mode):
    run = {"settings": {"start_year": 2002, "end_year": 2005, "initial_aum": 1000,
                        "factor_combination": mode},
           "factor_labels": ["ROA %", "ROE"], "factors": ["ROA", "roe"],
           "factor_directions": {"ROA": "bottom", "roe": "top"},
           "universe": {"restrict_fossil_fuels": True, "selected_sectors": ["Technology"]}}
    original = copy.deepcopy(run)
    snapshot = preview_for_run(run)
    assert "-Winsorize_1%(ROA %" in snapshot["formula"]
    assert "+Winsorize_1%(ROE" in snapshot["formula"]
    assert "ESG = exclude fossil fuel industries" in snapshot["formula"]
    assert run == original
    run["calculation_preview"] = snapshot
    run["settings"]["initial_aum"] = 9000
    assert preview_for_run(run) == snapshot
    assert "$1,000.00" in snapshot["formula"]


def test_preview_includes_new_trading_cost_setting():
    settings = {"start_year": 2010, "end_year": 2020, "initial_aum": 1000, "trading_cost_bps": 25}
    preview = build_calculation_preview(settings, ["ROA %"], {"ROA %": "top"})
    assert "Trading cost = 25 bps" in preview["formula"]
    assert "R_net(t)" in preview["formula"]
    assert preview["steps"][-1][0] == "Apply trading costs"


# ---------------------------------------------------------------------------
# The preview's numbers are the engine's: it must not drift from what it describes
# ---------------------------------------------------------------------------

def _engine_spy(monkeypatch):
    """Records the keyword arguments of every rebalance_portfolio call, local or in a Lambda job."""
    calls, real = [], engine.rebalance_portfolio

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(engine, "rebalance_portfolio", spy)
    monkeypatch.setattr(jobs, "rebalance_portfolio", spy)
    return calls


def _numbers(pattern, text):
    return [float(value) for value in re.findall(pattern, text)]


@pytest.mark.parametrize("path", ["local", "lambda"])
@pytest.mark.parametrize("chosen", [
    {},                                                                  # the defaults: 10%, no cost
    {"top_pct": 5.0, "trading_cost_bps": 25.0, "factor_combination": "integrating"},
    {"top_pct": 12.5, "trading_cost_bps": 7.5},
    {"top_pct": 33.3, "trading_cost_bps": 0.0, "factor_combination": "integrating"},
])
def test_preview_top_pct_and_trading_cost_match_the_engine(state, monkeypatch, path, chosen):
    calls = _engine_spy(monkeypatch)
    settings = {"start_year": 2002, "end_year": 2006, "initial_aum": 1000.0, "selected_sectors": [],
                "restrict_fossil_fuels": False, "use_lambda": path == "lambda", **chosen}
    if path == "lambda":
        class Client:
            def run(self, job):
                return run_job(state.raw_data, job), {"function_name": "test"}
        monkeypatch.setenv(utils.BACKTEST_LAMBDA_ENV, "test")
        monkeypatch.setattr(utils, "get_lambda_client", lambda name: Client())
    else:
        utils.load_and_process_data(settings)
    run = utils.run_backtest_logic(settings, ["ROA %", "ROE using 9/30 Data"],
                                   {"ROA %": "top", "ROE using 9/30 Data": "bottom"})
    assert len(calls) == 1
    engine = calls[0]
    preview = run["calculation_preview"]
    text = preview["formula"] + "\n" + "\n".join(f"{title}: {body}" for title, body in preview["steps"])
    # Every "highest X%" in the formula and its steps is the engine's top_pct
    shown = _numbers(r"highest ([\d.]+)%", text)
    assert len(shown) >= 2 and set(shown) == {float(engine["top_pct"])}
    # The trading cost lines show the engine's cost, and appear only when it charges one
    cost = float(engine["trading_cost_bps"])
    if cost > 0:
        assert _numbers(r"Trading cost = ([\d.]+) bps", text) == [cost]
        assert _numbers(r"c\(t\) = ([\d.]+) / 10000", text) == [cost]
        assert preview["steps"][-1][0] == "Apply trading costs"
    else:
        assert "Trading cost" not in text and "R_net" not in text
        assert "Apply trading costs" not in [title for title, _ in preview["steps"]]


@pytest.mark.parametrize("frequency, cost, hold", [
    ("Monthly", 0.0, None), ("Quarterly", 25.0, None), ("Semi-annual", 7.5, None),
    ("Quarterly", 0.0, ("Top X%", 25)), ("Monthly", 10.0, ("Top N stocks", 7)),
    ("Quarterly", 5.0, ("Top N stocks", 1)),
])
def test_wrds_preview_frequency_top_pct_and_trading_cost_match_the_wrds_engine(new_session, monkeypatch,
                                                                                 frequency, cost, hold):
    calls, real = [], service.run_wrds_backtest

    def spy(data, factors, directions, **options):
        calls.append(options)
        return real(data, factors, directions, **options)
    monkeypatch.setattr(service, "run_wrds_backtest", spy)    # what the Lambda's WRDS engine receives
    at = new_session()
    at.sidebar.number_input(key="wrds_start_year_input").set_value(2010).run()
    at.sidebar.number_input(key="wrds_end_year_input").set_value(2012).run()
    at.sidebar.radio(key="wrds_frequency_input").set_value(frequency).run()
    at.sidebar.number_input(key="trading_cost_bps_input").set_value(cost).run()
    if hold:                                                  # the Hold choice: Top X% or Top N stocks
        at.sidebar.radio(key="wrds_hold_input").set_value(hold[0]).run()
        key = "wrds_top_pct_input" if hold[0] == "Top X%" else "wrds_top_n_input"
        at.sidebar.number_input(key=key).set_value(hold[1]).run()
    for key in ("w_roe", "w_mom_12_1"):
        at.button(key=f"factor_add_{key}").click().run()
    _click_main_tab(at, "Analysis")
    analysis = next(c.value for c in at.main.code if c.value.startswith("Rebalance at month ends"))
    at.button(key="run_wrds_analysis").click().run()
    _assert_clean(at)
    run = at.session_state["saved_runs"][0]
    preview = run["calculation_preview"]
    assert preview["formula"] == analysis                    # Analysis showed the job the engine then received
    engine = calls[0]
    assert all({k: c[k] for k in ("frequency", "top_pct", "top_n", "trading_cost_bps")} ==
               {k: engine[k] for k in ("frequency", "top_pct", "top_n", "trading_cost_bps")} for c in calls)
    formula = preview["formula"]
    text = formula + "\n" + "\n".join(f"{title}: {body}" for title, body in preview["steps"])
    # Top % or top N: every selection rule and "top X%" / "top N stocks" is the engine's, never the other
    percents = _numbers(r"floor\(n [*×] ([\d.]+) / 100\)", text) + _numbers(r"top ([\d.]+)% by", text)
    counts = (_numbers(r"min\((\d+), n\)", text) + _numbers(r"top (\d+) stocks by", text)
              + _numbers(r"(\d+) different stocks?", text) + _numbers(r"Mixing, Top (\d+):", text))
    if engine["top_n"] is None:
        assert len(percents) >= 3 and set(percents) == {float(engine["top_pct"])} and not counts
        assert hold is None or engine["top_pct"] == hold[1]
    else:
        assert len(counts) >= 3 and set(counts) == {float(engine["top_n"])} and not percents
        assert engine["top_pct"] is None and engine["top_n"] == hold[1]
        # Top N holds only scored stocks, even when N covers the filter (only Top 100% holds the whole filter)
        assert "stocks without a score are never held" in formula and "every stock in U(t)" not in formula
        if engine["factor_combination"] == "mixing":       # N different stocks in total, the sleeves taking turns
            assert f"the sleeves take turns in factor order ({', '.join(run['factor_labels'])})" in formula
            assert "each sleeve gets V(t)" not in formula and "Across the picked stocks" in text
    # Frequency: its name, and as many rebalances as the engine made
    assert re.match(r"Rebalance at month ends, ([\w-]+) \(", formula).group(1) == FREQUENCY_LABELS[engine["frequency"]]
    assert preview["steps"][0][1].startswith(f"{FREQUENCY_LABELS[engine['frequency']]}: ")
    assert _numbers(r"\[(\d+) rebalances?;", formula) == [len(run["results"]["rebalance_dates"])]
    # Trading cost: the engine's, and its lines only when it charges one
    charged = float(engine["trading_cost_bps"])
    if charged > 0:
        assert _numbers(r"Trading cost = ([\d.]+) bps", text) == [charged]
        assert _numbers(r"c\(t\) = ([\d.]+) / 10,000", text) == [charged]
        assert _numbers(r"charge ([\d.]+) bps", text) == [charged]
    else:
        assert "Trading cost" not in text and "R_net" not in text
        assert "Apply trading costs" not in [title for title, _ in preview["steps"]]
