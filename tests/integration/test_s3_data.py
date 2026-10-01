"""
Integration test for the S3 data source (reads the real bucket).

Requires AWS credentials. Run with:
    AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... uv run pytest tests/integration/test_s3_data.py -v
"""
import os

import pytest

from src.data_standardization import REQUIRED_COLUMNS, validate_universe
from src.s3_client import S3DataManager

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.getenv('AWS_ACCESS_KEY_ID') or not os.getenv('AWS_SECRET_ACCESS_KEY'),
        reason="Requires AWS credentials (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"
    )
]


def test_universe_loads_from_s3():
    data = S3DataManager().fetch_all_data()

    validate_universe(data, 's3')
    assert len(data) > 10_000
    assert set(REQUIRED_COLUMNS) <= set(data.columns)
    assert data['Year'].min() <= 2002 and data['Year'].nunique() >= 20  # 2002 onward
    assert {'Delist_Date', 'Delist_Price'} <= set(data.columns)
