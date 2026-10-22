"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: aws_lambda/backtest_handler.py
PURPOSE: AWS Lambda entry point for backtests (handler: aws_lambda.backtest_handler.handler).
VERSION: 1.0.0

Deployed by scripts/deploy_backtest_lambda.py. The event is a job from
src/backtest_jobs.build_job (or {"action": "ping"}); see src/backtest_service.py
for the response.
"""

import logging
from typing import Any, Dict

from src.backtest_service import BacktestService

logging.getLogger().setLevel(logging.INFO)

# Created on the first invocation and kept while the container stays warm, so the
# market data is downloaded once per container rather than once per backtest.
_service = None


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    global _service
    if _service is None:
        _service = BacktestService()
    return _service.handle_event(event or {})
