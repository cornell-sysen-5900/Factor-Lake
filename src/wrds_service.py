"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: src/wrds_service.py
PURPOSE: Runs WRDS backtest jobs inside AWS Lambda on the WRDS data in S3, with results cached in S3.
VERSION: 2.0.0

src/backtest_service.py sends every job with data_mode 'wrds' here. For each job
(src/wrds_jobs.py) the service:
1. Hashes the job with the WRDS data version (the folder and the S3 ETags of the three
   files the engine reads) and the WRDS code version.
2. Returns the saved result for that hash when one exists. A saved backtest whose
   holdings file is missing is computed again, never shown without its holdings.
3. Otherwise runs the job and saves it (a backtest with its factor regression,
   src/factor_risk.py, in the result's factor_risk): a backtest's holdings first, as
   backtest-results/<hash>.holdings.parquet (one row group per rebalance date), then
   the result JSON (backtest-results/<hash>.json), which lists the holdings' dates.
   The result JSON therefore only exists once its holdings do. Its S3 metadata has the
   job's kind and a short summary, read by the list of saved analyses, and a backtest's
   scores (src/analysis_scores.py), read by the scoreboard.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .analysis_scores import score_metadata
from .backtest_jobs import encode_saved, job_hash
from .backtest_service import RESULTS_DIR, _error_code, _is_missing, result_file, summarize_job
from .factor_risk import wrds_factor_risk
from .s3_client import S3DataManager
from .wrds_calendar import FREQUENCY_LABELS
from .wrds_data import WRDS_FILES, WrdsData, read_s3_dollar_volume, read_s3_wrds_data, wrds_dir
from .wrds_engine import rank_stocks, run_wrds_backtest
from .wrds_holdings import holdings_file, holdings_parquet
from .wrds_jobs import (
    WRDS_TOP_PCT,
    describe_hold,
    describe_wrds_job,
    engine_arguments,
    validate_wrds_job,
    wrds_code_version,
)
from .wrds_liquidity import DAILY_MANIFEST, describe_liquidity_limit

logger = logging.getLogger(__name__)


def summarize_wrds_job(job: Dict[str, Any]) -> str:
    """
    A short description of a WRDS job for listings: the data source and frequency, then
    Legacy's summary, e.g. "WRDS, Quarterly | ROE (higher), ROA (higher) | Mixing |
    2002-2026 | $1,000", then how many stocks it holds when that is not the top 10%
    ("| Top 20 stocks") and its liquidity limit when it has one ("| Liquidity: Max 10% of
    daily volume"). Stored as S3 metadata, so it is kept ASCII and short.
    """
    frequency = FREQUENCY_LABELS.get(job['frequency'], job['frequency'])
    text = summarize_job(job)
    if not job['factors']:                 # a run holding the whole filter
        text = 'No factors' + text
    if job.get('top_n') is not None or job['top_pct'] != WRDS_TOP_PCT:
        text += f" | {describe_hold(job)}"
    if job.get('liquidity_limit') is not None:
        text += f" | Liquidity: {describe_liquidity_limit(job['liquidity_limit'])}"
    return f"WRDS, {frequency} | {text}".encode('ascii', 'replace').decode()[:500]


class WrdsBacktestRunner:
    """Loads the WRDS data from S3 once per data version and runs WRDS jobs on it."""

    def __init__(self, manager: Optional[S3DataManager] = None, data_dir: Optional[str] = None):
        self.manager = manager or S3DataManager()
        self.data_dir = data_dir or wrds_dir()
        self._data: Optional[Tuple[str, WrdsData]] = None   # (data version, data)
        self._volume: Optional[Tuple[str, pd.Series]] = None    # (daily manifest's ETag, dollar volume)

    def data_version(self) -> str:
        """
        Identifies the WRDS data a backtest reads: the folder and the S3 ETags of its
        three files. Republishing any of them changes it.
        """
        parts = [self.data_dir]
        for filename in WRDS_FILES:
            parts.append(f"{filename}:{self._etag(filename)}")
        return '|'.join(parts)

    def volume_version(self) -> str:
        """
        Identifies the daily prices the liquidity limit reads: the S3 ETag of their
        manifest, which is uploaded last whenever they are republished.
        """
        return f"{DAILY_MANIFEST}:{self._etag(DAILY_MANIFEST)}"

    def _etag(self, filename: str) -> str:
        head = self.manager.client.head_object(Bucket=self.manager.bucket,
                                               Key=self.manager.object_key(f'{self.data_dir}/{filename}'))
        return head['ETag'].strip(chr(34))

    def data(self, version: Optional[str] = None) -> WrdsData:
        """The WRDS data, reused while its version is unchanged."""
        version = version or self.data_version()
        if self._data is None or self._data[0] != version:
            self._data = None   # free the old copy before downloading the new one
            started = time.perf_counter()
            self._data = (version, read_s3_wrds_data(self.manager, self.data_dir))
            logger.info(f"Loaded the WRDS data ({self.data_dir}) in {time.perf_counter() - started:.1f} s")
        return self._data[1]

    def dollar_volume(self) -> pd.Series:
        """The month-end dollar volume of the liquidity limit, reused while the daily prices are unchanged."""
        version = self.volume_version()
        if self._volume is None or self._volume[0] != version:
            self._volume = None
            started = time.perf_counter()
            self._volume = (version, read_s3_dollar_volume(self.manager, self.data_dir))
            logger.info(f"Loaded the dollar volume ({self.data_dir}/prices_daily, {len(self._volume[1]):,} rows) "
                        f"in {time.perf_counter() - started:.1f} s")
        return self._volume[1]

    def run(self, job: Dict[str, Any], version: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Runs a WRDS job (src/wrds_jobs.py) on the WRDS data.

        Returns:
            backtest: (run_wrds_backtest results, holdings DataFrame). The results also have
                factor_risk: the factor regression of the monthly returns
                (src/factor_risk.wrds_factor_risk), None when the run has too few months.
            cohort: ({'top', 'bottom': portfolio values, 'years', 'cohort_pct', 'top_cagr',
                'bottom_cagr'}, None), the strategy at cohort_pct and its exact opposite.
            ranked: ({'rebalance_date', 'columns', 'rows'}, None), every stock on that date
                ranked (src/wrds_engine.rank_stocks).
        """
        validate_wrds_job(job)
        data = self.data(version)
        if job.get('liquidity_limit') is not None:      # only these jobs read the daily prices
            data.dollar_volume = self.dollar_volume()
        factors, directions, options = engine_arguments(job)
        if job['kind'] == 'backtest':
            results, holdings = run_wrds_backtest(data, factors, directions, **options)
            results['factor_risk'] = wrds_factor_risk(results, data.market)
            return results, holdings
        if job['kind'] == 'cohort':
            options.update(top_pct=job['cohort_pct'], top_n=None)
            top, _ = run_wrds_backtest(data, factors, directions, **options)
            inverse = {f: 'bottom' if d == 'top' else 'top' for f, d in directions.items()}
            bottom, _ = run_wrds_backtest(data, factors, inverse, **options)
            return {'top': top['portfolio_values'], 'bottom': bottom['portfolio_values'], 'years': top['years'],
                    'cohort_pct': job['cohort_pct'], 'top_cagr': top['cagr'], 'bottom_cagr': bottom['cagr']}, None
        date = data.month_date(job['rebalance_date'])
        table = rank_stocks(data, date, factors, directions, job['factor_combination'], job['top_pct'],
                            job['use_market_cap_weight'], job['restrict_fossil_fuels'], job['selected_sectors'],
                            job.get('gics_filter'), job.get('top_n'))
        return {'rebalance_date': date.strftime('%Y-%m-%d'), 'columns': list(table.columns),
                'rows': table.to_dict('records')}, None


class WrdsBacktestService:
    """
    Runs and caches WRDS jobs. Created by BacktestService (one per Lambda container), so
    it shares its S3 access, result cache, inline limit and response format.
    """

    def __init__(self, service, runner: Optional[WrdsBacktestRunner] = None):
        self.service = service
        self.manager = service.manager
        self.runner = runner or WrdsBacktestRunner(self.manager)
        self.code_version = wrds_code_version()

    def run(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """The job's result, from the cache when possible (see the module docstring)."""
        validate_wrds_job(job)
        base_version = self.runner.data_version()
        # A job with a liquidity limit also reads the daily prices; other jobs hash as before
        data_version = (f"{base_version}|{self.runner.volume_version()}" if job.get('liquidity_limit') is not None
                        else base_version)
        key = job_hash(job, data_version, self.code_version)
        filename = result_file(key)

        cached = self.service._read_cached(filename)
        if cached is not None and job['kind'] == 'backtest' and not self._exists(holdings_file(RESULTS_DIR, key)):
            logger.warning(f"Holdings of {key[:12]} are missing; computing the result again")
            cached = None
        if cached is not None:
            body, info = cached
            logger.info(f"Cache hit {key[:12]} for {describe_wrds_job(job)}")
        else:
            started = time.perf_counter()
            result, holdings = self.runner.run(job, base_version)
            if holdings is not None:
                result['holdings'] = self._save_holdings(key, holdings)
            info = {
                'computed_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'compute_seconds': f"{time.perf_counter() - started:.3f}",
                # Read by the list of saved analyses without downloading each result
                'kind': job['kind'],
                'summary': summarize_wrds_job(job),
                # Read by the scoreboard, to rank analyses without downloading them
                **score_metadata(job, result),
            }
            body = encode_saved(job, result)   # with its job, so the hash alone can reopen it
            self._put(filename, body, 'application/json', info)
            logger.info(f"Computed {key[:12]} in {info['compute_seconds']} s ({len(body)} bytes) "
                        f"for {describe_wrds_job(job)}")
        return self.service._respond(key, filename, body, info, cached is not None, data_version,
                                     self.code_version)

    def _save_holdings(self, key: str, holdings: pd.DataFrame) -> Dict[str, Any]:
        """Writes the holdings Parquet and returns what the result JSON says about it."""
        body, layout = holdings_parquet(holdings)
        filename = holdings_file(RESULTS_DIR, key)
        self._put(filename, body, 'application/vnd.apache.parquet')
        return {'file': filename, 'bytes': len(body), **layout}

    def _put(self, filename: str, body: bytes, content_type: str, metadata: Optional[Dict[str, str]] = None) -> None:
        self.manager.client.put_object(Bucket=self.manager.bucket, Key=self.manager.object_key(filename),
                                       Body=body, ContentType=content_type, Metadata=metadata or {})

    def _exists(self, filename: str) -> bool:
        """
        True when the object exists. Without s3:ListBucket, S3 answers 403 instead of 404
        for a missing object, so that counts as missing too.
        """
        try:
            self.manager.client.head_object(Bucket=self.manager.bucket, Key=self.manager.object_key(filename))
            return True
        except Exception as e:
            if _is_missing(e) or _error_code(e) in ('AccessDenied', '403', 'Forbidden'):
                return False
            raise
