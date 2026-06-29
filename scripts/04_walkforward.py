"""
scripts/04_walkforward.py — Out-Of-Sample (OOS) reporting

Slices full backtest outputs from ``scripts/03_run_backtest.py`` to the held-out
tail defined by ``CONFIG.oos_fraction``. If full outputs are missing, runs
``run_backtest(CONFIG)`` once to generate them (same continuous simulation as 03).

Usage:
    uv run python scripts/03_run_backtest.py
    uv run python scripts/04_walkforward.py

Outputs:
    outputs/backtest_results/oos_trade_log.csv
    outputs/backtest_results/oos_nav_series.csv
    outputs/backtest_results/oos_pair_daily_mtm.csv (when pair_daily_mtm.csv exists)
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_backtest
from src.config import CONFIG
from src.data.load import load_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _compute_oos_start(
    all_dates: list[pd.Timestamp],
    oos_fraction: float,
) -> tuple[int, pd.Timestamp]:
    """
    Compute the first trading index and timestamp in the OOS tail.

    Args:
        all_dates: Sorted unique trading timestamps from prices.
        oos_fraction: Fraction of trailing calendar to treat as OOS (e.g. 0.3).

    Returns:
        Tuple ``(cutoff_idx, oos_start)`` where ``oos_start = all_dates[cutoff_idx]``.
    """
    cutoff_idx = int(len(all_dates) * (1.0 - oos_fraction))
    cutoff_idx = max(0, min(cutoff_idx, len(all_dates) - 1))
    return cutoff_idx, all_dates[cutoff_idx]


def _slice_trades_oos(
    trade_log: pd.DataFrame,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return trade rows with ``date`` on or after ``oos_start``.

    Args:
        trade_log: Full-sample trade log from the engine.
        oos_start: First OOS trading timestamp (inclusive).

    Returns:
        Filtered trade log (possibly empty).
    """
    if trade_log.empty:
        return trade_log.copy()
    if "date" not in trade_log.columns:
        raise ValueError("trade_log is missing required column 'date'")
    dates = pd.to_datetime(trade_log["date"], utc=False)
    return trade_log.loc[dates >= pd.Timestamp(oos_start)].copy()


def _slice_pair_daily_oos(
    pair_daily: pd.DataFrame,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return pair EOD rows with ``date`` on or after ``oos_start``.

    Args:
        pair_daily: Full-sample ``pair_daily_mtm`` from the engine.
        oos_start: First OOS trading timestamp (inclusive).

    Returns:
        Filtered DataFrame (possibly empty).
    """
    if pair_daily.empty:
        return pair_daily.copy()
    if "date" not in pair_daily.columns:
        raise ValueError("pair_daily_mtm is missing required column 'date'")
    dates = pd.to_datetime(pair_daily["date"], utc=False)
    return pair_daily.loc[dates >= pd.Timestamp(oos_start)].copy()


def _slice_nav_oos(
    nav_series: pd.DataFrame,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return NAV rows with ``date`` on or after ``oos_start``.

    Args:
        nav_series: Full-sample NAV series from the engine.
        oos_start: First OOS trading timestamp (inclusive).

    Returns:
        Filtered NAV series (possibly empty).
    """
    if nav_series.empty:
        return nav_series.copy()
    if "date" not in nav_series.columns:
        raise ValueError("nav_series is missing required column 'date'")
    dates = pd.to_datetime(nav_series["date"], utc=False)
    return nav_series.loc[dates >= pd.Timestamp(oos_start)].copy()


def main() -> None:
    """
    Build OOS trade log and NAV series from the full backtest outputs.

    Reads ``outputs/backtest_results/trade_log.csv``, ``nav_series.csv``, and
    ``pair_daily_mtm.csv`` when present; otherwise runs ``run_backtest(CONFIG)``
    once, writes those files, then slices the OOS tail using ``CONFIG.oos_fraction``.

    Args:
        None

    Returns:
        None
    """
    logger.info("=" * 60)
    logger.info("  ARQ PAIRS TRADING — OUT-OF-SAMPLE SLICE")
    logger.info("=" * 60)

    start_date = (
        pd.Timestamp(CONFIG.backtest_start_date).date()
        if CONFIG.backtest_start_date
        else None
    )
    end_date = (
        pd.Timestamp(CONFIG.backtest_end_date).date()
        if CONFIG.backtest_end_date
        else None
    )
    all_prices = load_prices(start=start_date, end=end_date)
    all_dates = sorted(all_prices["date"].unique())
    cutoff_idx, oos_start = _compute_oos_start(all_dates, CONFIG.oos_fraction)

    if cutoff_idx > 0:
        logger.info(
            "Last in-sample trading day (reporting boundary): %s",
            all_dates[cutoff_idx - 1].date(),
        )
    else:
        logger.info("OOS slice starts at first trading day (cutoff_idx=0).")
    logger.info("OOS evaluation starts: %s", oos_start.date())

    out_dir = Path("outputs/backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    trade_path_full = out_dir / "trade_log.csv"
    nav_path_full = out_dir / "nav_series.csv"
    pair_path_full = out_dir / "pair_daily_mtm.csv"

    if trade_path_full.exists() and nav_path_full.exists():
        logger.info("Loading full backtest results from %s", out_dir)
        trade_log = pd.read_csv(trade_path_full)
        nav_series = pd.read_csv(nav_path_full)
        pair_daily = (
            pd.read_csv(pair_path_full)
            if pair_path_full.exists()
            else pd.DataFrame()
        )
        if pair_daily.empty and not pair_path_full.exists():
            logger.warning(
                "pair_daily_mtm.csv missing — re-run scripts/03 after upgrading the engine."
            )
    else:
        logger.info(
            "Full backtest outputs not found — running run_backtest(CONFIG) once..."
        )
        trade_log, nav_series, pair_daily = run_backtest(CONFIG)
        trade_log.to_csv(trade_path_full, index=False)
        nav_series.to_csv(nav_path_full, index=False)
        pair_daily.to_csv(pair_path_full, index=False)

    oos_trades = _slice_trades_oos(trade_log, oos_start)
    oos_nav = _slice_nav_oos(nav_series, oos_start)
    oos_pair = _slice_pair_daily_oos(pair_daily, oos_start)

    trade_path = out_dir / "oos_trade_log.csv"
    nav_path = out_dir / "oos_nav_series.csv"
    pair_path = out_dir / "oos_pair_daily_mtm.csv"
    oos_trades.to_csv(trade_path, index=False)
    oos_nav.to_csv(nav_path, index=False)
    oos_pair.to_csv(pair_path, index=False)

    logger.info("=" * 60)
    logger.info("  OUT-OF-SAMPLE SLICE COMPLETE")
    logger.info("=" * 60)
    logger.info("  OOS trade rows        : %d", len(oos_trades))
    if not oos_nav.empty:
        start_nav = float(oos_nav["nav"].iloc[0])
        end_nav = float(oos_nav["nav"].iloc[-1])
        oos_return = (end_nav - start_nav) / start_nav if start_nav else 0.0
        logger.info("  OOS Starting NAV      : $%.2f", start_nav)
        logger.info("  OOS Final NAV         : $%.2f", end_nav)
        logger.info("  OOS total return      : %.2f%%", oos_return * 100.0)
    logger.info("  OOS Trade Log         : %s", trade_path)
    logger.info("  OOS NAV Series        : %s", nav_path)
    logger.info("  OOS pair MTM rows     : %d → %s", len(oos_pair), pair_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
