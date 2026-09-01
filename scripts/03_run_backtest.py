"""
scripts/03_run_backtest.py — Main Backtest Execution

Executes the ARQ pairs trading backtest over the full configured date range.
The engine runs one continuous simulation; script 04 slices OOS metrics using
``CONFIG.oos_fraction``.

Usage:
    uv run python scripts/03_run_backtest.py

Outputs:
    outputs/backtest_results/trade_log.csv
    outputs/backtest_results/nav_series.csv
    outputs/backtest_results/pair_daily_mtm.csv
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

    Uses parameters from ``src/config.py``. The final ``oos_fraction`` tail is
    not truncated here; use ``scripts/04_walkforward.py`` for OOS CSV slices.
    
    Args:
        None

    Returns:
        None
    """
    logger.info("=" * 60)
    logger.info("  ARQ PAIRS TRADING — FULL BACKTEST")
    logger.info("=" * 60)
    
    # Run the backtest over the full configured date range (single continuous NAV path).
    logger.info(
        "Running full-calendar backtest (oos_fraction=%.2f used only by script 04 for slicing)...",
        CONFIG.oos_fraction,
    )
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
