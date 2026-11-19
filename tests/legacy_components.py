"""Isolated regression harness for the retained Legacy components.

This is NOT the production entry point. The WRDS-only app is exercised separately
by test_wrds_preview_app, test_wrds_saved, test_share_links and test_compare_tab.
Legacy calculation/rendering tests can still drive their old components without
restoring a Legacy launch path in the shipped application.
"""
from streamlit.testing.v1 import AppTest


def legacy_components_app():
    from unittest.mock import patch

    import streamlit as st

    import streamlit_app as app
    from app.calculation_preview import build_calculation_preview
    from components.data_viewer import render_legacy_data_viewer_tab
    from components.factor_selection import render_factor_selection

    app.utils.initialize_environment()
    app.utils.initialize_session_state()
    app.css.inject_styles()

    def legacy_mode(backtest_lambda, on_change):
        # Test inputs for the retained sidebar's Legacy branch, never shipped.
        st.session_state['data_mode_input'] = 'Legacy'
        return False

    def legacy_run(settings, factors, directions):
        app._render_legacy_run(settings, app.utils.lambda_backtests_enabled(settings), factors, directions)

    with (patch.object(app.sidebar, '_render_data_mode', legacy_mode),
          patch.object(app, 'render_factor_selection', lambda _: render_factor_selection('legacy')),
          patch.object(app, '_render_wrds_run', legacy_run),
          patch.object(app, 'wrds_analysis_preview', build_calculation_preview),
          patch.object(app, 'render_data_viewer_tab', render_legacy_data_viewer_tab)):
        app.main()


def legacy_app_test():
    return AppTest.from_function(legacy_components_app, default_timeout=60)
