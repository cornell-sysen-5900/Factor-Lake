"""Live, data-independent explanations of the strategy a backtest runs.

Two formulas, one per engine, never mixed: build_calculation_preview describes the Legacy
engine (annual rebalance on Next-Years_Return, its delisting options, FactSet industry
exclusions); build_wrds_calculation_preview describes the WRDS engine (src/wrds_engine.py)
from the WRDS job it receives. A WRDS run never shows the Legacy formula.
"""
import calendar
from typing import Any, Dict, List, Optional


def build_calculation_preview(settings: Dict[str, Any], factors: List[str],
                              directions: Dict[str, str]) -> Dict[str, Any]:
    """Capture a self-contained formula; never retain mutable settings or widget state."""
    count = len(factors)
    pct = float(settings.get("top_pct", 10.0))
    start, end = int(settings["start_year"]), int(settings["end_year"])
    sectors = list(settings.get("selected_sectors") or [])
    sector_text = ", ".join(sectors) if sectors else "all sectors (no sector restriction)"
    esg = "exclude fossil fuel industries" if settings.get("restrict_fossil_fuels") else "no fossil fuel exclusion"
    lines = [f"U = market data with Year in [{start}, {end}]; sectors = {sector_text}; ESG = {esg}"]
    terms = []
    for index, name in enumerate(factors, 1):
        sign = "+" if directions.get(name, "top") == "top" else "-"
        tilt = "Higher is better" if sign == "+" else "Lower is better"
        lines.append(f"z{index}(i,t) = ZScore({sign}Winsorize_1%({name}(i,t)))  [{tilt}]")
        terms.append(f"z{index}")
    combined = " + ".join(terms) if terms else "[select at least one factor]"
    integrating = settings.get("factor_combination", "mixing") == "integrating"
    if integrating:
        lines += [f"Integrating: Score(i,t) = ({combined}) / {count or 'K'}",
                  f"H(t) = highest {pct:g}% of complete-case Score in U(t)",
                  "A(t) = V(t)  [one combined portfolio]"]
    else:
        sleeves = " + ".join(f"Sleeve[{name}]" for name in factors) or "[select at least one factor]"
        lines += [f"Mixing: Portfolio(t) = ({sleeves}); capital per sleeve = V(t) / {count or 'K'}",
                  f"Each H_k(t) = highest {pct:g}% of valid z_k in U(t)",
                  f"A_k(t) = V(t) / {count or 'K'}"]
    if settings.get("use_market_cap_weight"):
        lines.append("Market Cap Weight: w_i = MarketCap_i / sum(MarketCap_j for j in H)")
        weighting_step = ("Within each selected portfolio, weight by market capitalization. Missing caps become zero; "
                          "if the cap column is absent or the sum is not positive, use equal weights.")
        lines.append("Fallback: w_i = 1 / |H| when market caps are absent or their sum is not positive")
    else:
        lines.append("Equal Weight: w_i = 1 / |H|")
        weighting_step = "Within each selected portfolio, allocate the same amount to each selected stock."
    lines += ["Position_i = A * w_i; shares_i = Position_i / Ending_Price_i (positive valid prices only)"]
    strategy = settings.get("delisting_strategy", "zero_return")
    post_rate, label = {
        "zero_return": ("0", "Zero Return"),
        "hold_cash": ("risk-free rate for year t", "Hold Cash"),
        "reinvest": ("value-weighted return of surviving positions (0 if none)", "Reinvest"),
    }[strategy]
    lines += [f"Delisting = {label}; post-delist rate q = {post_rate}",
              "r_i* = r_pre + (1 + r_pre) * q * remaining_year_fraction  [delisted positions]",
              "R(t) = sum(Position_i * r_i*) / sum(Position_i)  [all purchased positions]",
              f"V({start}) = ${float(settings['initial_aum']):,.2f}; V(t+1) = V(t) * (1 + R(t))",
              f"Rebalance annually for t = {start}, ..., {end - 1}"]
    steps = [
        ("Filter the universe", f"Use years {start}–{end}; sectors: {sector_text}; ESG: {esg}. "
         "The fossil fuel filter uses the application's FactSet industry exclusions when industry data is available."),
        ("Normalize each selected factor", "Cap each factor at its 1st and 99th percentiles within each filtered year, "
         "then apply the selected direction (+ for higher, − for lower). Subtract its mean and divide by the sample "
         "standard deviation. For a constant factor, center without dividing. Non-finite values are missing."),
        ("Select stocks and combine factors",
         ("Average all selected factor z-scores; exclude stocks missing any selected factor. Buy the highest "
          f"{pct:g}% once in one portfolio." if integrating else
          f"Create one sleeve per factor, each buying its highest {pct:g}% of valid z-scores. Divide capital "
          "equally among sleeves. Stocks selected by multiple factors occur in multiple sleeves.") +
         " Select max(1, floor(eligible count × percentage / 100)) when eligible stocks exist. "
         "A positive, valid formation price is required to buy a selected stock; "
         "skipped allocations are not redistributed."),
        ("Allocate capital", weighting_step + " A denotes the capital of the portfolio or sleeve; "
         "H is its selected stock set. "
         "The return calculation normalizes by the value of positions actually purchased."),
        ("Handle delisted positions", f"{label}: keep the price return earned before delisting, "
         f"then apply q = {post_rate} "
         "to capital at delisting for the remaining days / 365. If the delisting price is unavailable, r_pre is 0; "
         "if its date is unavailable, the remaining-year fraction is 0.5. Live positions use Next-Years_Return / 100. "
         "Here r_i* is the effective decimal return of a live or delisted position."),
        ("Compound portfolio value", f"Start with ${float(settings['initial_aum']):,.2f}. "
         "Combine all purchased positions' "
         f"value-weighted returns, then compound annually from {start} to {end}. Years without data are skipped; "
         "a year with no purchased positions has a zero return. K is the number of selected factors."),
    ]
    cost_bps = float(settings.get("trading_cost_bps") or 0)
    if cost_bps > 0:
        lines += [f"Trading cost = {cost_bps:g} bps per buy or sell",
                  "Turnover(t) = sum(abs(target_weight_i - drifted_weight_i))",
                  f"c(t) = {cost_bps:g} / 10000 * Turnover(t)",
                  "R_net(t) = R(t) - c(t) * (1 + R(t)); V(t+1) = V(t) * (1 + R_net(t))"]
        steps.append(("Apply trading costs", "Charge buys and sells at each rebalance based on turnover, "
                      "starting from cash on the first rebalance. Compound net returns after costs; "
                      "there is no final liquidation charge."))
    return {"formula": "\n".join(lines), "steps": steps}


def build_wrds_calculation_preview(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    The WRDS engine's calculation for one WRDS job (src/wrds_jobs.build_wrds_job): every
    number and choice is the job's, every rule is src/wrds_engine.py's. A job may have no
    factors yet (the Analysis tab before any is added). Raises ValueError for years
    without a complete holding period, like the job itself.
    """
    from src.trading_costs import BPS, job_trading_cost
    from src.wrds_calendar import FIRST_MONTH, FREQUENCY_LABELS, LAST_MONTH, REBALANCE_MONTHS, period_boundaries
    from src.wrds_engine import BENCHMARK_LABELS, FOSSIL_GSUBIND, FOSSIL_SIC, RISK_FREE_SOURCE, WINSORIZE_PCT
    from src.wrds_gics import describe_gics_filter, job_gics_filter, level_label, level_names

    frequency, start, end = job["frequency"], int(job["start_year"]), int(job["end_year"])
    months = period_boundaries(frequency, start, end)
    label = FREQUENCY_LABELS[frequency]
    month_names = ("every month" if len(REBALANCE_MONTHS[frequency]) == 12 else
                   ", ".join(calendar.month_abbr[m] for m in REBALANCE_MONTHS[frequency]))
    count = len(months) - 1
    rebalances = f"{count} rebalance{'s' if count != 1 else ''}"
    pct, low = float(job["top_pct"]), WINSORIZE_PCT * 100
    factors = list(job["factors"])
    k = len(factors)
    aum = float(job["initial_aum"])
    cost_bps = job_trading_cost(job)

    # a. Universe: the GICS filter (level and choices) and the fossil fuel filter
    gics = job_gics_filter(job.get("gics_filter"), job.get("selected_sectors"))
    if not gics:
        gics_line, gics_step = "GICS filter = none", "No GICS filter."
    elif gics["level"] == "sector":
        gics_line = (f"GICS filter = Sector: {describe_gics_filter(gics)}  "
                     "[sector_name; a missing GICS code is filled from SIC]")
        gics_step = (f"GICS filter at the Sector level: {describe_gics_filter(gics)}. It reads sector_name, which "
                     "fills a missing GICS code from the stock's SIC code, so only stocks without any sector "
                     "(Unclassified) are left out.")
    else:
        level = level_label(gics["level"])
        names = ", ".join(sorted(level_names(gics["level"]).get(c, str(c)) for c in gics["codes"]))
        gics_line = f"GICS filter = {describe_gics_filter(gics)}  [GICS sub-industry starts with a chosen code]"
        gics_step = (f"GICS filter at the {level} level: {names}. A stock is kept when its "
                     f"GICS sub-industry code starts with a chosen {level.lower()} code; stocks without GICS "
                     "are left out.")
    gsubind, sic = ", ".join(map(str, FOSSIL_GSUBIND)), ", ".join(map(str, FOSSIL_SIC))
    if job["restrict_fossil_fuels"]:
        fossil_line = f"Fossil fuel filter = exclude GICS sub-industries {gsubind}; without GICS, SIC {sic}"
        fossil_step = (f"Fossil fuel filter: leaves out GICS sub-industries {gsubind} and, for stocks without "
                       f"GICS, SIC codes {sic}.")
    else:
        fossil_line, fossil_step = "Fossil fuel filter = off", "No fossil fuel filter."

    lines = [f"Rebalance at month ends, {label} ({month_names}): t = {months[0]}, ..., {months[-2]}  "
             f"[{rebalances}; the last period ends {months[-1]}]",
             "U(t) = Russell 2000 members at t in the WRDS panel with price > 0 and market cap > 0",
             gics_line, fossil_line]
    # b. Scores
    terms = []
    for index, column in enumerate(factors, 1):
        higher = job["factor_directions"].get(column, "top") == "top"
        name = job["factor_labels"].get(column, column)
        lines.append(f"z{index}(i,t) = ZScore_ddof1({'+' if higher else '-'}Winsorize_{low:g}%/{100 - low:g}%"
                     f"({name}(i,t)))  [{'Higher' if higher else 'Lower'} is better; within date t]")
        terms.append(f"z{index}")
    combined = ", ".join(terms) if terms else "[select at least one factor]"
    # c. Selection and weights
    top = f"max(1, floor(n * {pct:g} / 100))"
    integrating = job["factor_combination"] == "integrating"
    if integrating:
        lines += [f"Integrating: Score(i,t) = mean({combined}) over stocks with all {k or 'K'} factors",
                  f"H(t) = top {top} of the n stocks with a Score, by Score; ties by lower permno",
                  "W_i(t) = w_i  [one portfolio with all of V(t)]"]
    else:
        sleeves = " + ".join(f"Sleeve[{job['factor_labels'].get(c, c)}]" for c in factors)
        lines += [f"Mixing: Portfolio(t) = ({sleeves or '[select at least one factor]'}); "
                  f"each sleeve gets V(t) / {k or 'K'}",
                  f"H_k(t) = top {top} of the n stocks with a z_k, by z_k; ties by lower permno",
                  f"W_i(t) = sum over the sleeves k holding i of w_ik / {k or 'K'}"]
    if job["use_market_cap_weight"]:
        lines.append("Market Cap Weight: w_i = mkt_cap_musd_i(t) / sum(mkt_cap_musd_j(t) for j in H)")
        weight_step = "Within each selection, weight by market capitalization at t (mkt_cap_musd)."
    else:
        lines.append("Equal Weight: w_i = 1 / |H|")
        weight_step = "Within each selection, give every stock the same weight."
    # d. Holding periods, e. trading costs, f. benchmarks
    lines += ["g_i = prod over months m in (t, t'] of (1 + r_i,m)  "
              "[CRSP monthly total return, delisting return included]",
              "After i's delisting month: (1 + rf_m) instead; a missing month before any delisting: 0%",
              "R(t') = sum(W_i(t) * g_i) - 1  [buy and hold; no delisting option]; "
              "nothing to buy at t: R(t') = rf over (t, t']"]
    if cost_bps > 0:
        lines += [f"Trading cost = {cost_bps:g} bps per buy or sell",
                  "Turnover(t) = sum(abs(target_weight_i - drifted_weight_i))  [delisted money is cash]",
                  f"c(t) = {cost_bps:g} / {BPS:,.0f} * Turnover(t)",
                  "R_net(t') = R(t') - c(t) * (1 + R(t')); V(t') = V(t) * (1 + R_net(t'))"]
    else:
        lines.append("V(t') = V(t) * (1 + R(t'))")
    lines += [f"V({months[0]}) = ${aum:,.2f}",
              f"Benchmark = {BENCHMARK_LABELS['benchmark']}; also {BENCHMARK_LABELS['growth']} and "
              f"{BENCHMARK_LABELS['value']}, over the same periods",
              f"rf = {RISK_FREE_SOURCE}"]

    steps = [
        ("Set the rebalance dates",
         f"{label}: rebalance at the month ends of {month_names} (the panel's last CRSP trading day of the month). "
         f"The run covers the calendar years {start}–{end}: it first rebalances in December {start - 1} and its "
         f"last period ends in December {end}, kept to whole periods within the data ({FIRST_MONTH} to "
         f"{LAST_MONTH}). Here: {rebalances}, {months[0]} to {months[-2]}; holds to {months[-1]}."),
        ("Filter the universe",
         "At each date t: the Russell 2000 members known at t in the WRDS panel, with a positive price and "
         f"market cap. Nothing dated after t is read. {gics_step} {fossil_step}"),
        ("Normalize each selected factor",
         f"Within each date: non-numbers become missing; cap the values at the {low:g}% and {100 - low:g}% "
         "quantiles; negate when lower is better; subtract the mean and divide by the standard deviation "
         "(ddof=1). A date whose values do not vary is only centered."),
        ("Select stocks and combine factors",
         (f"Average the z-scores of the stocks that have every factor and buy the top {pct:g}% by that score "
          "in one portfolio." if integrating else
          f"One sleeve per factor, each with an equal share of the money, buying the top {pct:g}% by its "
          "z-score. A stock picked by several sleeves is held in each.") +
         f" Each selection is max(1, floor(n × {pct:g} / 100)) of the n stocks with a score, best first, "
         "ties broken by lower permno."),
        ("Allocate capital", weight_step),
        ("Hold to the next rebalance",
         "Buy and hold over (t, t']: weights drift with each stock's compounded CRSP monthly total returns, "
         "which include delisting returns, so there is no delisting option. After a stock's delisting month "
         f"its money earns the risk-free rate until t' (rf: {RISK_FREE_SOURCE}). A month without a return before "
         "any delisting counts as 0% and is counted in the results. A date with nothing to buy holds cash "
         "at the risk-free rate."),
    ]
    if cost_bps > 0:
        steps.append(("Apply trading costs",
                      f"At each rebalance, charge {cost_bps:g} bps on the turnover: the sum of |target weight − "
                      "weight after drift| over all stocks. Money from a delisted stock is cash, so reinvesting "
                      "it is a charged buy; the first rebalance buys from cash (100% turnover). The cost is paid "
                      "before the period's return; there is no final sell."))
    steps.append(("Compound and compare",
                  f"Start with ${aum:,.2f} and compound each period's return"
                  f"{' after costs' if cost_bps > 0 else ''}. The benchmark is {BENCHMARK_LABELS['benchmark']} "
                  f"over the same periods (outperformance is against it); {BENCHMARK_LABELS['growth']} and "
                  f"{BENCHMARK_LABELS['value']} are shown too."))
    return {"mode": "wrds", "formula": "\n".join(lines), "steps": steps}


def wrds_preview(job: Dict[str, Any]) -> Dict[str, Any]:
    """build_wrds_calculation_preview, or a placeholder with the reason the job's years cannot run."""
    try:
        return build_wrds_calculation_preview(job)
    except ValueError as error:
        return {"mode": "wrds", "formula": f"[{error}]", "steps": []}


def wrds_analysis_preview(settings: Dict[str, Any], factor_names: List[str],
                          directions: Dict[str, str]) -> Dict[str, Any]:
    """
    The Analysis tab's WRDS preview: the calculation of the job the Run button would send.
    Before any factor is added, the same job without factors.
    """
    from app.streamlit_config import WRDS_FACTOR_METADATA
    from app.streamlit_utils import wrds_job_for_inputs
    names = [f for f in factor_names if f in WRDS_FACTOR_METADATA]
    try:
        if names:
            return build_wrds_calculation_preview(wrds_job_for_inputs(settings, names, directions))
        job = wrds_job_for_inputs(settings, [next(iter(WRDS_FACTOR_METADATA))], {})
        return build_wrds_calculation_preview({**job, "factors": [], "factor_directions": {}, "factor_labels": {}})
    except ValueError as error:
        return {"mode": "wrds", "formula": f"[{error}]", "steps": []}


def preview_for_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    The formula a run used: its saved copy, or rebuilt from its saved inputs (older and
    opened analyses), never from the current sidebar. A WRDS run always gets the WRDS
    formula, never the Legacy one.
    """
    stored = run.get("calculation_preview")
    if run.get("data_mode") == "wrds":
        if stored and stored.get("mode") == "wrds":
            return stored
        from app.streamlit_utils import wrds_run_job
        return wrds_preview(wrds_run_job(run))
    if stored:
        return stored
    settings = dict(run["settings"])
    settings.update(run.get("universe") or {})
    labels = run["factor_labels"]
    directions = {label: run["factor_directions"].get(column, "top")
                  for label, column in zip(labels, run["factors"])}
    return build_calculation_preview(settings, labels, directions)


# The collapsed expanders that hold a preview: in Analysis, and a run's in Results
PREVIEW_LABEL, SAVED_LABEL = "Calculation preview", "Calculation used in this run"
STEPS_LABEL = "Step-by-step calculation"


def render_calculation_preview(preview: Dict[str, Any], *, saved: bool = False) -> None:
    """
    The formula and its steps in one expander, collapsed so it does not compete with the
    factors (Analysis) or the results (Results); the steps are a nested expander.
    """
    import streamlit as st
    with st.expander(SAVED_LABEL if saved else PREVIEW_LABEL, expanded=False):
        st.caption("Saved run inputs; changing current selections does not change this formula." if saved else
                   "Updates with your selections. Data is not required; "
                   "numerical scores are computed when you run the analysis.")
        st.code(preview["formula"], language="text", wrap_lines=True)
        if not preview["steps"]:
            return
        with st.expander(STEPS_LABEL, expanded=False):
            for number, (title, description) in enumerate(preview["steps"], 1):
                st.markdown(f"**{number}. {title}**")
                st.write(description)
