"""
scripts/03_run_backtest.py — Main Backtest Execution

Executes the ARQ pairs trading backtest over the full configured date range.
The engine runs one continuous simulation. Script 04 charts these CSVs.

Usage:
    uv run python scripts/03_run_backtest.py

Outputs:
    outputs/backtest_results/trade_log.csv
    outputs/backtest_results/nav_series.csv
    outputs/backtest_results/pair_daily_mtm.csv
    outputs/backtest_results/blocked_entries.csv
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_backtest
from src.config import CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Execute the full-calendar backtest and write trade_log, nav_series, and
    per-pair EOD MTM snapshots (``pair_daily_mtm.csv``).

    Uses parameters from ``src/config.py``. After this script finishes, run
    ``scripts/04_generate_report.py`` for charts and a metrics summary.
    
    Args:
        None

    Returns:
        None
    """
    logger.info("=" * 60)
    logger.info("  ARQ PAIRS TRADING — FULL BACKTEST")
    logger.info("=" * 60)
    
    logger.info("Running full-calendar backtest...")
    trade_log, nav_series, pair_daily, blocked_entries = run_backtest(CONFIG)

    # Prepare output directories
    out_dir = Path("outputs/backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    trade_path = out_dir / "trade_log.csv"
    nav_path = out_dir / "nav_series.csv"
    pair_path = out_dir / "pair_daily_mtm.csv"
    blocked_path = out_dir / "blocked_entries.csv"

    trade_log.to_csv(trade_path, index=False)
    nav_series.to_csv(nav_path, index=False)
    pair_daily.to_csv(pair_path, index=False)
    blocked_entries.to_csv(blocked_path, index=False)
    
    logger.info("=" * 60)
    logger.info("  BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total Trades : {len(trade_log)}")
    logger.info(f"  Final NAV    : ${nav_series['nav'].iloc[-1]:,.2f}" if not nav_series.empty else "  Final NAV    : $0.00")
    logger.info(f"  Trade Log    : {trade_path}")
    logger.info(f"  NAV Series   : {nav_path}")
    logger.info(f"  Pair MTM EOD : {pair_path} ({len(pair_daily)} rows)")
    logger.info(f"  Blocked      : {blocked_path} ({len(blocked_entries)} rows)")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
