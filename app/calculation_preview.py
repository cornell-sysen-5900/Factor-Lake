"""Live, data-independent explanations of the strategy used by the backtest."""
from typing import Any, Dict, List


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
         "A positive, valid formation price is required to buy a selected stock; skipped allocations are not redistributed."),
        ("Allocate capital", weighting_step + " A denotes the capital of the portfolio or sleeve; H is its selected stock set. "
         "The return calculation normalizes by the value of positions actually purchased."),
        ("Handle delisted positions", f"{label}: keep the price return earned before delisting, then apply q = {post_rate} "
         "to capital at delisting for the remaining days / 365. If the delisting price is unavailable, r_pre is 0; "
         "if its date is unavailable, the remaining-year fraction is 0.5. Live positions use Next-Years_Return / 100. "
         "Here r_i* is the effective decimal return of a live or delisted position."),
        ("Compound portfolio value", f"Start with ${float(settings['initial_aum']):,.2f}. Combine all purchased positions' "
         f"value-weighted returns, then compound annually from {start} to {end}. Years without data are skipped; "
         "a year with no purchased positions has a zero return. K is the number of selected factors."),
    ]
    return {"formula": "\n".join(lines), "steps": steps}


def preview_for_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Use saved inputs for older/opened analyses, never the current sidebar."""
    if run.get("calculation_preview"):
        return run["calculation_preview"]
    settings = dict(run["settings"])
    settings.update(run.get("universe") or {})
    labels = run["factor_labels"]
    directions = {label: run["factor_directions"].get(column, "top")
                  for label, column in zip(labels, run["factors"])}
    return build_calculation_preview(settings, labels, directions)


def render_calculation_preview(preview: Dict[str, Any], *, saved: bool = False) -> None:
    """Render the formula without loading data; steps are expandable."""
    import streamlit as st
    with st.container(border=True):
        st.subheader("Calculation Used in This Run" if saved else "Calculation Preview")
        st.caption("Saved run inputs; changing current selections does not change this formula." if saved else
                   "Updates with your selections. Data is not required; numerical scores are computed when you run the analysis.")
        st.code(preview["formula"], language="text", wrap_lines=True)
        with st.expander("Step-by-step calculation", expanded=False):
            for number, (title, description) in enumerate(preview["steps"], 1):
                st.markdown(f"**{number}. {title}**")
                st.write(description)
