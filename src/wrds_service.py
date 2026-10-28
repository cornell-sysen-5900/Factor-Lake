"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_service.py
PURPOSE: Runs WRDS backtests inside AWS Lambda on the WRDS data in S3.
VERSION: 1.0.0

One runner lives for the life of a Lambda container, so warm invocations reuse the
WRDS data already in memory. The job schema, the result cache and the dispatch from
src/backtest_service.py come in stage 4 (DOCS/WRDS_MODE_DESIGN.md, section 5).
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .s3_client import S3DataManager
from .wrds_data import WRDS_FILES, WrdsData, read_s3_wrds_data, wrds_dir
from .wrds_engine import run_wrds_backtest

logger = logging.getLogger(__name__)

# Job keys the runner needs, and the defaults of the optional ones
REQUIRED_JOB_KEYS = ('factors', 'factor_directions', 'frequency', 'start_year', 'end_year', 'initial_aum')
JOB_DEFAULTS = {'top_pct': 10.0, 'use_market_cap_weight': False, 'factor_combination': 'mixing',
                'restrict_fossil_fuels': False, 'selected_sectors': None, 'factor_labels': None}


class WrdsBacktestRunner:
    """Loads the WRDS data from S3 once per data version and runs WRDS backtests on it."""

    def __init__(self, manager: Optional[S3DataManager] = None, data_dir: Optional[str] = None):
        self.manager = manager or S3DataManager()
        self.data_dir = data_dir or wrds_dir()
        self._data: Optional[Tuple[str, WrdsData]] = None   # (data version, data)

    def data_version(self) -> str:
        """
        Identifies the WRDS data a backtest reads: the folder and the S3 ETags of its
        three files. Republishing any of them changes it.
        """
        parts = [self.data_dir]
        for filename in WRDS_FILES:
            head = self.manager.client.head_object(Bucket=self.manager.bucket,
                                                   Key=self.manager.object_key(f'{self.data_dir}/{filename}'))
            parts.append(f"{filename}:{head['ETag'].strip(chr(34))}")
        return '|'.join(parts)

    def data(self, version: Optional[str] = None) -> WrdsData:
        """The WRDS data, reused while its version is unchanged."""
        version = version or self.data_version()
        if self._data is None or self._data[0] != version:
            self._data = None   # free the old copy before downloading the new one
            started = time.perf_counter()
            self._data = (version, read_s3_wrds_data(self.manager, self.data_dir))
            logger.info(f"Loaded the WRDS data ({self.data_dir}) in {time.perf_counter() - started:.1f} s")
        return self._data[1]

    def run(self, job: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs a WRDS backtest job: the run_wrds_backtest arguments as a dict.

        Returns:
            (results, holdings) from run_wrds_backtest.

        Raises:
            ValueError: a missing key or an input the engine rejects.
        """
        if missing := [k for k in REQUIRED_JOB_KEYS if k not in job]:
            raise ValueError(f"The WRDS backtest request is missing {missing}.")
        options = {k: job.get(k, default) for k, default in JOB_DEFAULTS.items()}
        return run_wrds_backtest(
            self.data(), list(job['factors']), dict(job['factor_directions']), job['frequency'],
            int(job['start_year']), int(job['end_year']), float(job['initial_aum']), **options)
