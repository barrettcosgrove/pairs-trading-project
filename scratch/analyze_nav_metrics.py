# scratch/analyze_nav_metrics.py
#
# Point-in-time performance stats from nav_series.csv with a user-chosen
# analysis start (same idea as scratch/plot.py: edit one line and re-run).
#
# Usage:
#   uv run python scratch/analyze_nav_metrics.py
#
# Edit ANALYSIS_START_MODE / ANALYSIS_* below — no CLI required.

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# User knobs — change how the analysis window starts
# ---------------------------------------------------------------------------
# Mode "offset_days": start = first CSV date + OFFSET_DAYS (matches plot.py style).
# Mode "fixed_date":  start = FIXED_START_DATE (must be on/after first row).
# Mode "full":        use entire series from first row (no trim).
ANALYSIS_START_MODE: str = "offset_days"  # "offset_days" | "fixed_date" | "full"

# Used when ANALYSIS_START_MODE == "offset_days"
OFFSET_DAYS: int = 800

# Used when ANALYSIS_START_MODE == "fixed_date" (string or None to disable)
FIXED_START_DATE: str | None = "2023-03-15"

# Annualized risk-free rate for excess-return Sharpe (set 0.0 to use raw returns)
RISK_FREE_ANNUAL: float = 0.04

# Path to NAV series (same default as scratch/plot.py)
NAV_CSV = Path(__file__).parent.parent / "outputs" / "backtest_results" / "nav_series.csv"


def resolve_start_date(
    index: pd.DatetimeIndex,
    mode: str,
    offset_days: int,
    fixed_start: str | None,
) -> pd.Timestamp:
    """
    Return the first timestamp to include in the analysis window (inclusive).

    Args:
        index: Sorted datetime index of the NAV series.
        mode: One of ``offset_days``, ``fixed_date``, or ``full``.
        offset_days: Calendar days after ``index.min()`` when mode is offset_days.
        fixed_start: ISO date string when mode is fixed_date.

    Returns:
        Inclusive start timestamp for filtering.

    Raises:
        ValueError: If mode is unknown or fixed_start is invalid for the mode.
    """
    first = pd.Timestamp(index.min()).normalize()
    last = pd.Timestamp(index.max()).normalize()

    if mode == "full":
        return first

    if mode == "offset_days":
        start = first + timedelta(days=offset_days)
        if start > last:
            raise ValueError(
                f"offset_days={offset_days} pushes start ({start.date()}) "
                f"after last data ({last.date()})."
            )
        return start

    if mode == "fixed_date":
        if not fixed_start:
            raise ValueError("FIXED_START_DATE must be set when mode is fixed_date.")
        start = pd.Timestamp(fixed_start).normalize()
        if start < first:
            raise ValueError(
                f"FIXED_START_DATE {start.date()} is before first data row {first.date()}."
            )
        if start > last:
            raise ValueError(
                f"FIXED_START_DATE {start.date()} is after last data row {last.date()}."
            )
        return start

    raise ValueError(f"Unknown ANALYSIS_START_MODE: {mode!r}")


def max_drawdown_from_nav(nav: pd.Series) -> float:
    """
    Maximum drawdown as a positive fraction (e.g. 0.15 = 15% peak-to-trough).

    Uses a running peak on NAV within the provided slice only.

    Args:
        nav: NAV levels indexed by date, ascending.

    Returns:
        Max drawdown in [0, 1] if NAV is positive; 0.0 if empty or invalid.
    """
    nav = nav.astype(float).dropna()
    if len(nav) < 2:
        return 0.0
    if (nav <= 0).any():
        return float("nan")
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(-dd.min())


def annualized_sharpe(
    daily_returns: pd.Series,
    risk_free_annual: float = 0.0,
) -> float:
    """
    Annualized Sharpe from daily simple returns (252 trading days).

    Args:
        daily_returns: Daily simple returns (e.g. from NAV pct_change).
        risk_free_annual: Annual risk-free rate; converted to daily and subtracted.

    Returns:
        Sharpe ratio, or NaN if insufficient data or zero volatility.
    """
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_d = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    excess = r - rf_d
    std = float(excess.std())
    if std < 1e-12 or std != std:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(252.0))


def summarize_window(nav_df: pd.DataFrame) -> dict[str, float | str]:
    """
    Compute total return %, max drawdown, and annualized Sharpe for a window.

    Args:
        nav_df: Subset of the NAV CSV with index ``date`` and column ``nav``.

    Returns:
        Dict of printable metrics including date range strings.
    """
    nav = nav_df["nav"].sort_index()
    if nav.empty:
        raise ValueError("Empty window — widen the date range or check the CSV.")

    r = nav.pct_change()
    start_nav = float(nav.iloc[0])
    end_nav = float(nav.iloc[-1])
    total_return_pct = (end_nav / start_nav - 1.0) * 100.0 if start_nav else float("nan")
    mdd = max_drawdown_from_nav(nav)
    sharpe = annualized_sharpe(r, RISK_FREE_ANNUAL)

    return {
        "start_date": str(nav.index.min().date()),
        "end_date": str(nav.index.max().date()),
        "n_days": int(len(nav)),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(mdd * 100.0) if mdd == mdd else float("nan"),
        "sharpe_annual": float(sharpe) if sharpe == sharpe else float("nan"),
    }


def main() -> None:
    """Load NAV CSV, apply start filter, print metrics."""
    df = pd.read_csv(NAV_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    start = resolve_start_date(
        df.index,
        ANALYSIS_START_MODE,
        OFFSET_DAYS,
        FIXED_START_DATE,
    )
    window = df[df.index >= start].copy()

    metrics = summarize_window(window)

    print("NAV metrics (analysis window)")
    print("  CSV          :", NAV_CSV)
    print("  mode         :", ANALYSIS_START_MODE)
    print("  start (incl.):", metrics["start_date"])
    print("  end          :", metrics["end_date"])
    print("  trading rows :", metrics["n_days"])
    print("  total return : {:.2f}%".format(metrics["total_return_pct"]))
    print(
        "  max drawdown : {:.2f}% (within-window peak NAV)".format(
            metrics["max_drawdown_pct"]
        )
    )
    print(
        "  Sharpe (ann.): {:.3f} (daily NAV returns, rf={:.1%} p.a.)".format(
            metrics["sharpe_annual"],
            RISK_FREE_ANNUAL,
        )
    )


if __name__ == "__main__":
    main()
