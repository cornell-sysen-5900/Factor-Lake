"""Start the synthetic-data demo: uv run python scripts/run_demo.py (port 8502)."""
import os
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment['FACTOR_LAKE_DEMO'] = '1'
    environment['FACTOR_LAKE_BACKTEST_LAMBDA'] = ''
    raise SystemExit(subprocess.call([
        sys.executable, '-m', 'streamlit', 'run', str(root / 'app/streamlit_app.py'),
        '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8502',
        '--browser.gatherUsageStats=false'], cwd=root, env=environment))
