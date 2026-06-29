# scratch/spy_buyhold_returns.py
#
# Buy-and-hold SPY total return over a date window, using the same raw regime
# file as the rest of the project (SPY column from data/raw/regime.parquet).
# Run from the repository root after fetch (scripts/01_fetch_data.py).
#
# Usage:
#   uv run python scratch/spy_buyhold_returns.py
#
# Edit ANALYSIS_START_DATE / ANALYSIS_END_DATE below (inclusive, YYYY-MM-DD).

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGIME_PATH = REPO_ROOT / "data" / "raw" / "regime.parquet"

# ---------------------------------------------------------------------------
# User knobs — change the window here, then re-run the script
# ---------------------------------------------------------------------------
ANALYSIS_START_DATE: str = "2019-01-01"
ANALYSIS_END_DATE: str = "2024-12-31"


def load_spy_adj_close(start: date, end: date) -> pd.Series:
    """
    Load daily SPY adjusted close from the pipeline regime parquet.

    Args:
        start: First calendar date to include (inclusive).
        end: Last calendar date to include (inclusive).

    Returns:
        Series of adjusted closes indexed by date (normalized to midnight UTC),
        sorted ascending, with no NaNs in the returned slice.

    Raises:
        FileNotFoundError: If regime.parquet is missing.
        ValueError: If the SPY column is missing or the filtered series is empty.
    """
    if not REGIME_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {REGIME_PATH}. Run the data fetch step first so regime "
            "data (VIX + SPY) exists."
        )

    df = pd.read_parquet(REGIME_PATH)
    if "spy" not in df.columns:
        raise ValueError(
            f"{REGIME_PATH} has no 'spy' column (expected merged regime layout)."
        )

    work = df[["date", "spy"]].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    work = work[(work["date"] >= ts_start) & (work["date"] <= ts_end)]
    work = work.dropna(subset=["spy"])

    if work.empty:
        raise ValueError(
            f"No SPY rows between {start} and {end}. Check dates and regime file coverage."
        )

    series = work.set_index("date")["spy"].sort_index()
    series.name = "SPY"
    return series.astype(float)


def summarize_buy_and_hold(close: pd.Series) -> dict[str, object]:
    """
    Compute total and annualized buy-and-hold metrics from a price level series.

    Args:
        close: Daily adjusted closes, sorted ascending by date.

    Returns:
        Dictionary with first/last timestamps, trading-day count, total simple
        and log return, simple CAGR (252-day year), annualized log-return
        volatility, and max drawdown on a normalized price path.
    """
    close = close.dropna()
    if len(close) < 2:
        raise ValueError("Need at least two prices to compute a return.")

    p0 = float(close.iloc[0])
    p1 = float(close.iloc[-1])
    log_ret = np.log(close / close.shift(1)).dropna()

    total_simple = p1 / p0 - 1.0
    total_log = float(np.log(p1 / p0))

    n = len(close)
    n_ret = len(log_ret)
    years_from_returns = n_ret / 252.0
    cagr_simple = (
        (1.0 + total_simple) ** (1.0 / years_from_returns) - 1.0
        if years_from_returns > 0
        else float("nan")
    )

    vol_log_annual = float(log_ret.std(ddof=1) * np.sqrt(252)) if n_ret > 1 else float("nan")

    wealth = close.astype(float) / p0
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    max_drawdown = float(dd.min())

    return {
        "first_date": close.index[0],
        "last_date": close.index[-1],
        "n_trading_days": n,
        "total_simple_return": total_simple,
        "total_log_return": total_log,
        "cagr_simple": cagr_simple,
        "vol_log_annual": vol_log_annual,
        "max_drawdown": max_drawdown,
    }


def main() -> int:
    """
    Entry point: load SPY, print summary statistics.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = date.fromisoformat(ANALYSIS_START_DATE)
    end = date.fromisoformat(ANALYSIS_END_DATE)

    try:
        spy = load_spy_adj_close(start=start, end=end)
        stats = summarize_buy_and_hold(spy)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    print("SPY buy-and-hold (from data/raw/regime.parquet, 'spy' column)")
    print(f"  Window: {stats['first_date'].date()} -> {stats['last_date'].date()}")
    print(f"  Trading days (levels): {int(stats['n_trading_days'])}")
    print(f"  Total simple return: {stats['total_simple_return']:.4%}")
    print(f"  Total log return:    {stats['total_log_return']:.4%}")
    print(f"  CAGR (simple, from trading-day count): {stats['cagr_simple']:.4%}")
    print(f"  Ann. vol (daily log returns):         {stats['vol_log_annual']:.4%}")
    print(f"  Max drawdown (on close-to-close path): {stats['max_drawdown']:.4%}")
    print()
    print(
        "Compare to your strategy using the same calendar window "
        "(e.g. NAV or PnL from outputs/backtest_results)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
