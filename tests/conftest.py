import pytest


@pytest.fixture(autouse=True)
def backtests_run_locally_by_default(monkeypatch):
    """
    App tests run backtests in-process, even when a developer's .env sets
    FACTOR_LAKE_BACKTEST_LAMBDA (the app copies .env into the environment only for
    variables that are not set yet, so an empty value wins). Tests of the Lambda path
    set it themselves.
    """
    monkeypatch.setenv('FACTOR_LAKE_BACKTEST_LAMBDA', '')
