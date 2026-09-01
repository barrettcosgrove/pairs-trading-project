"""
scripts/04_generate_report.py — Charts and metrics from backtest CSVs

Reads outputs written by scripts/03_run_backtest.py and writes the core chart
pack plus a text summary. Does not re-run the engine.

Usage:
    uv run python scripts/04_generate_report.py

Inputs (from scripts/03_run_backtest.py):
    outputs/backtest_results/trade_log.csv
    outputs/backtest_results/nav_series.csv
    outputs/backtest_results/blocked_entries.csv

Outputs:
    outputs/report/nav_and_drawdown.png
    outputs/report/monthly_returns_heatmap.png
    outputs/report/exit_type_mix.png
    outputs/report/blocked_entries.png
    outputs/report/metrics_summary.txt
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.performance import compute
from src.metrics.reporting import format_metrics_summary, generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("outputs/backtest_results")
REPORT_DIR = Path("outputs/report")
REQUIRED_FILES = ("trade_log.csv", "nav_series.csv", "blocked_entries.csv")


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load a backtest CSV, returning an empty frame if the file is empty.

    Args:
        path: CSV path written by script 03.

    Returns:
        DataFrame (possibly empty).
    """
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> int:
    """
    Load script 03 CSVs, write report artifacts, and print the metrics summary.

    Args:
        None

    Returns:
        Process exit code: 0 on success, 1 if script 03 outputs are missing.
    """
    missing = [name for name in REQUIRED_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        logger.error(
            "Missing %s under %s — run scripts/03_run_backtest.py first.",
            ", ".join(missing),
            RESULTS_DIR,
        )
        return 1

    trade_log = _load_csv(RESULTS_DIR / "trade_log.csv")
    nav_series = _load_csv(RESULTS_DIR / "nav_series.csv")
    blocked_entries = _load_csv(RESULTS_DIR / "blocked_entries.csv")

    logger.info("=" * 60)
    logger.info("  ARQ PAIRS TRADING — REPORT")
    logger.info("=" * 60)

    written = generate_report(trade_log, nav_series, blocked_entries, REPORT_DIR)
    metrics = compute(trade_log, nav_series)
    print(format_metrics_summary(metrics), end="")

    logger.info("Wrote %d artifact(s) to %s", len(written), REPORT_DIR)
    for name in written:
        logger.info("  %s", name)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
