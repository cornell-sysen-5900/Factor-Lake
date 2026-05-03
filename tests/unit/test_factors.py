"""
Test suite for factor_registry.py and factor_utils.py
Tests registry completeness, column resolution, and signal normalization.
"""
import pytest
import pandas as pd
import numpy as np
from src.factor_registry import FACTOR_REGISTRY, get_factor_column
from src.factor_utils import normalize_series


class TestFactorRegistry:
    """Test the FACTOR_REGISTRY dict and get_factor_column lookup."""

    def test_registry_not_empty(self):
        """Registry should contain factor mappings."""
        assert len(FACTOR_REGISTRY) > 0

    def test_all_keys_are_strings(self):
        """All registry keys should be strings."""
        for key in FACTOR_REGISTRY:
            assert isinstance(key, str)

    def test_all_values_are_strings(self):
        """All registry values (column names) should be strings."""
        for val in FACTOR_REGISTRY.values():
            assert isinstance(val, str)

    def test_known_factors_present(self):
        """Core factors expected by the backtest engine should be registered."""
        expected_keys = [
            'momentum_12m', 'momentum_6m', 'momentum_1m',
            'roe', 'roa', 'ptb', 'fey', 'vol',
            'accruals', 'roa_pct', 'asset_growth', 'capex_growth', 'btp',
        ]
        for key in expected_keys:
            assert key in FACTOR_REGISTRY, f"Missing registry key: {key}"

    def test_known_column_mappings(self):
        """Spot-check that key factors resolve to the correct DB column."""
        assert FACTOR_REGISTRY['momentum_6m'] == '6-Mo_Momentum'
        assert FACTOR_REGISTRY['roe'] == 'ROE_using_9-30_Data'
        assert FACTOR_REGISTRY['roa'] == 'ROA_using_9-30_Data'
        assert FACTOR_REGISTRY['vol'] == '1-Yr_Price_Vol'

    def test_get_factor_column_registered(self):
        """get_factor_column should return the mapped column for known keys."""
        assert get_factor_column('momentum_6m') == '6-Mo_Momentum'
        assert get_factor_column('roe') == 'ROE_using_9-30_Data'

    def test_get_factor_column_passthrough(self):
        """Unregistered keys should pass through unchanged (used when the
        caller already has the DB column name)."""
        assert get_factor_column('6-Mo_Momentum') == '6-Mo_Momentum'
        assert get_factor_column('SomeUnknownFactor') == 'SomeUnknownFactor'

    def test_no_duplicate_column_names(self):
        """Each DB column should only be mapped once."""
        values = list(FACTOR_REGISTRY.values())
        assert len(values) == len(set(values)), "Duplicate column names in FACTOR_REGISTRY"


class TestNormalizeSeries:
    """Test normalize_series from factor_utils.py."""

    def test_basic_zscore(self):
        """Normalized output should be mean-zero, unit-variance."""
        s = pd.Series([10, 20, 30, 40, 50])
        result = normalize_series(s, higher_is_better=True)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.std(ddof=1) == pytest.approx(1.0, abs=1e-10)

    def test_higher_is_better_true(self):
        """When higher_is_better=True, higher raw values should yield higher scores."""
        s = pd.Series([10, 20, 30])
        result = normalize_series(s, higher_is_better=True)
        assert result.iloc[2] > result.iloc[0]

    def test_higher_is_better_false(self):
        """When higher_is_better=False, lower raw values should yield higher scores."""
        s = pd.Series([10, 20, 30])
        result = normalize_series(s, higher_is_better=False)
        assert result.iloc[0] > result.iloc[2]

    def test_all_nan_returns_nan(self):
        """All-NaN input should return all-NaN output."""
        s = pd.Series([np.nan, np.nan, np.nan])
        result = normalize_series(s)
        assert result.isna().all()

    def test_nan_values_preserved(self):
        """NaN values should stay NaN; non-NaN values should be normalized."""
        s = pd.Series([10, np.nan, 30, 40])
        result = normalize_series(s)
        assert result.isna().sum() == 1
        assert pd.isna(result.iloc[1])

    def test_inf_values_become_nan(self):
        """Infinite values should be coerced to NaN."""
        s = pd.Series([10, np.inf, 30, -np.inf])
        result = normalize_series(s)
        assert not np.isinf(result.dropna()).any()

    def test_single_value(self):
        """A single non-NaN value should not raise."""
        s = pd.Series([42.0])
        result = normalize_series(s)
        assert len(result) == 1

    def test_constant_series(self):
        """Constant series has zero variance — should not raise."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = normalize_series(s)
        # All values identical → std=0, so z-score just centers at 0
        assert (result == 0.0).all()

    def test_winsorization_clips_outliers(self):
        """Extreme outliers should be clipped by winsorization."""
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])
        result_no_win = normalize_series(s, winsorize_pct=None)
        result_win = normalize_series(s, winsorize_pct=0.05)
        # With winsorization, the max normalized value should be smaller
        assert result_win.max() < result_no_win.max()

    def test_no_zscore(self):
        """With zscore=False, output should preserve relative magnitudes."""
        s = pd.Series([10, 20, 30])
        result = normalize_series(s, zscore=False, winsorize_pct=None)
        assert result.iloc[2] > result.iloc[0]
        # Should not be standardized
        assert result.mean() != pytest.approx(0.0, abs=0.01)

    def test_reciprocal_method(self):
        """Reciprocal inversion should flip rankings without negation."""
        s = pd.Series([1, 2, 4, 8])
        result = normalize_series(s, higher_is_better=False, method='reciprocal')
        # After reciprocal, 1 → 1.0, 8 → 0.125, so 1 should score highest
        assert result.iloc[0] > result.iloc[3]
