"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: app/components/factor_selection.py
PURPOSE: UI component for building the factor list and choosing each factor's direction (tilt).
VERSION: 3.0.0
"""

import streamlit as st
import streamlit_config as config
from typing import Dict, List, Tuple

HIGHER = "Higher is better"
LOWER = "Lower is better"
DIRECTION_OPTIONS = [LOWER, HIGHER]
DIRECTION_HELP = (
    "Higher is better: buy stocks with the highest values. "
    "Lower is better: buy stocks with the lowest values."
)

# The backtest buys the highest values for 'top' and the lowest for 'bottom'
# (higher_is_better = direction == 'top' in src/backtest_engine.py).
DIRECTION_TO_ENGINE = {HIGHER: 'top', LOWER: 'bottom'}
ENGINE_TO_DIRECTION = {'top': HIGHER, 'bottom': LOWER}

# Session state: {UI label: 'top' | 'bottom'}, in the order the factors were added
SELECTED_KEY = 'factor_builder_selected'
# Bumped after each Add so the "Add factors" panel is a new, closed and empty widget
ADD_NONCE_KEY = 'factor_builder_add_nonce'

# The old checkbox grid returned factors in this group order (left column, then
# right). Returning the same order keeps run labels, Results captions and the
# backtest's summation order exactly as before.
_OUTPUT_GROUP_ORDER = ['Momentum', 'Profitability', 'Growth', 'Value', 'Quality']


def render_factor_selection() -> Tuple[List[str], Dict[str, str]]:
    """
    Renders the factor builder: an "Add factors" panel grouped by category,
    one row per added factor with its direction control, and a summary of the
    strategy.

    Returns:
        Tuple[List[str], Dict[str, str]]: A list of selected factor names and
            a dictionary mapping those names to their ranking direction ('top' or 'bottom').
    """
    st.session_state.setdefault(SELECTED_KEY, {})
    st.session_state.setdefault(ADD_NONCE_KEY, 0)
    selected: Dict[str, str] = st.session_state[SELECTED_KEY]

    st.header("Factor Selection")
    st.write("Add the factors you want, then choose which direction the portfolio should favor.")

    _render_add_factors(selected)

    if not selected:
        st.caption("No factors added yet. Choose one above.")
    for name in list(selected):
        _render_factor_row(name, selected)

    st.write("---")

    if selected:
        factor_labels = [f"{name} ({ENGINE_TO_DIRECTION[d]})" for name, d in selected.items()]
        st.success(f"Selected {len(selected)} factor(s): {', '.join(factor_labels)}")
    else:
        st.warning("Please select at least one factor to run the analysis")

    st.write("---")

    selected_names = sorted(selected, key=_output_sort_key)
    return selected_names, {name: selected[name] for name in selected_names}


def _render_add_factors(selected: Dict[str, str]) -> None:
    """
    Dropdown panel listing the factors not yet added under their category titles.
    Several factors can be picked at once; Add appends them with their default directions.
    """
    nonce = st.session_state[ADD_NONCE_KEY]
    remaining = {
        group: [name for name, meta in config.FACTOR_METADATA.items()
                if meta['group'] == group and name not in selected]
        for group in config.FACTOR_GROUPS
    }
    with st.popover(
        "Add factors",
        icon=":material/add:",
        disabled=not any(remaining.values()),
        help=None if any(remaining.values()) else "All factors are added.",
        key=f"factor_add_popover_{nonce}",
    ):
        st.caption("Pick one or more factors, then click Add.")
        chosen: List[str] = []
        for group, names in remaining.items():
            if not names:
                continue
            picked = st.pills(
                f"**{group}**",
                options=names,
                selection_mode="multi",
                # Pills have no per-option tooltip, so the group help lists its definitions
                help="\n\n".join(f"**{n}**: {config.FACTOR_METADATA[n]['tooltip']}" for n in names),
                key=f"factor_pick_{config.FACTOR_GROUPS.index(group)}_{nonce}",
            )
            chosen += picked
        label = f"Add {len(chosen)} factor(s)" if chosen else "Add"
        if st.button(label, type="primary", disabled=not chosen, key="factor_add"):
            for name in chosen:
                default = HIGHER if config.FACTOR_METADATA[name]['higher_is_better'] else LOWER
                selected[name] = DIRECTION_TO_ENGINE[default]
                # Seed the row's direction control with the factor's default
                st.session_state[_direction_key(name)] = default
            st.session_state[ADD_NONCE_KEY] += 1
            st.rerun()


def _render_factor_row(name: str, selected: Dict[str, str]) -> None:
    """One added factor: name with tooltip, group beneath, direction control, remove button."""
    meta = config.FACTOR_METADATA[name]
    dir_key = _direction_key(name)
    # Re-seed the control from the stored direction if Streamlit dropped its state
    if dir_key not in st.session_state:
        st.session_state[dir_key] = ENGINE_TO_DIRECTION[selected[name]]

    name_col, dir_col, remove_col = st.columns([3, 3, 1], vertical_alignment="center")
    with name_col:
        st.markdown(f"**{name}**", help=meta['tooltip'])
        st.caption(meta['group'])
    with dir_col:
        st.segmented_control(
            "Direction",
            options=DIRECTION_OPTIONS,
            key=dir_key,
            help=DIRECTION_HELP,
            on_change=_on_direction_change,
            args=(name, dir_key),
        )
    with remove_col:
        if st.button("", icon=":material/close:", help="Remove", key=f"factor_remove_{meta['key']}"):
            del selected[name]
            st.rerun()


def _on_direction_change(name: str, dir_key: str) -> None:
    """Stores the new direction; clicking the active option (deselect) keeps the old one."""
    selected = st.session_state[SELECTED_KEY]
    choice = st.session_state[dir_key]
    if choice is None:
        st.session_state[dir_key] = ENGINE_TO_DIRECTION[selected[name]]
    else:
        selected[name] = DIRECTION_TO_ENGINE[choice]


def _direction_key(name: str) -> str:
    return f"factor_dir_{config.FACTOR_METADATA[name]['key']}"


def _output_sort_key(name: str) -> Tuple[int, int]:
    """Position of a factor in the old checkbox grid order."""
    group = config.FACTOR_METADATA[name]['group']
    return _OUTPUT_GROUP_ORDER.index(group), config.FACTOR_OPTIONS.index(name)
