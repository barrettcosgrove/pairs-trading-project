"""
src/metrics/performance.py — Backtest summary metrics

Computes return, risk, and trade-mix statistics from the full-sample NAV
series and trade log written by scripts/03_run_backtest.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

EXIT_ACTIONS = frozenset(
    {
        "TAKE_PROFIT",
        "STOP_LOSS",
        "PLATEAU_STOP",
        "TIME_STOP",
        "EARNINGS_EXIT",
        "DOLLAR_STOP",
    }
)

PNL_COLUMN_CANDIDATES = ("realized_net_usd", "pnl")


def _exit_rows(trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Return trade-log rows whose action is an exit.

    Args:
        trade_log: Full trade log, possibly empty.

    Returns:
        Subset of ``trade_log`` containing only exit actions.
    """
    if trade_log.empty or "action" not in trade_log.columns:
        return trade_log.iloc[0:0].copy()
    return trade_log.loc[trade_log["action"].isin(EXIT_ACTIONS)].copy()


def _pnl_series(exits: pd.DataFrame) -> pd.Series:
    """
    Choose the best available P&L column on exit rows.

    Prefers ``realized_net_usd`` (net of costs) and falls back to ``pnl``.

    Args:
        exits: Exit-only trade log.

    Returns:
        Numeric P&L series aligned to ``exits``. Empty series if neither column
        is present.
    """
    for column in PNL_COLUMN_CANDIDATES:
        if column in exits.columns:
            return pd.to_numeric(exits[column], errors="coerce")
    return pd.Series(dtype=float)


def _max_drawdown(nav_series: pd.DataFrame) -> float:
    """
    Maximum drawdown as a positive fraction of peak NAV.

    Uses ``drawdown_from_peak`` when present; otherwise recomputes from NAV.

    Args:
        nav_series: NAV frame with at least a ``nav`` column.

    Returns:
        Max drawdown in [0, 1], or 0.0 if it cannot be measured.
    """
    if "drawdown_from_peak" in nav_series.columns:
        values = pd.to_numeric(nav_series["drawdown_from_peak"], errors="coerce")
        if values.notna().any():
            return float(values.max())

    nav = pd.to_numeric(nav_series["nav"], errors="coerce").dropna()
    if len(nav) < 2 or (nav <= 0).any():
        return 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(-dd.min())


def _annualized_ratio(daily_returns: pd.Series, downside_only: bool) -> float:
    """
    Annualized Sharpe (or Sortino) from daily simple returns with rf = 0.

    Args:
        daily_returns: Daily NAV pct_change values.
        downside_only: If True, divide by downside deviation (Sortino).

    Returns:
        Annualized ratio, or NaN when volatility is zero or the series is short.
    """
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    if downside_only:
        downside = r.clip(upper=0.0)
        std = float(np.sqrt((downside ** 2).mean()))
    else:
        std = float(r.std())
    if std < 1e-12 or math.isnan(std):
        return float("nan")
    return float(r.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def compute(trade_log: pd.DataFrame, nav_series: pd.DataFrame) -> dict:
    """
    Compute full-sample performance metrics from a backtest trade log and NAV.

    Sharpe and Sortino use daily NAV returns with a zero risk-free rate.

    Args:
        trade_log: Engine trade log. Exit actions feed win rate and mix stats.
        nav_series: Daily NAV frame with columns ``date`` (optional) and ``nav``.

    Returns:
        Dict of scalar metrics. Missing inputs yield NaN or empty nested dicts
        rather than raising.
    """
    metrics: dict = {
        "start_nav": float("nan"),
        "end_nav": float("nan"),
        "n_days": 0,
        "total_return": float("nan"),
        "annualized_return": float("nan"),
        "sharpe": float("nan"),
        "sortino": float("nan"),
        "max_drawdown": float("nan"),
        "n_exits": 0,
        "n_wins": 0,
        "win_rate": float("nan"),
        "exit_pnl_net": float("nan"),
        "mean_days_open": float("nan"),
        "mean_expected_halflife": float("nan"),
        "exit_counts": {},
        "exit_pnl": {},
    }

    if nav_series is None or nav_series.empty or "nav" not in nav_series.columns:
        nav = pd.Series(dtype=float)
    else:
        ordered = nav_series.copy()
        if "date" in ordered.columns:
            ordered = ordered.sort_values("date")
        nav = pd.to_numeric(ordered["nav"], errors="coerce").dropna()

    if not nav.empty:
        start_nav = float(nav.iloc[0])
        end_nav = float(nav.iloc[-1])
        n_days = int(len(nav))
        metrics["start_nav"] = start_nav
        metrics["end_nav"] = end_nav
        metrics["n_days"] = n_days
        if start_nav:
            total_return = end_nav / start_nav - 1.0
            metrics["total_return"] = float(total_return)
            if n_days > 1:
                metrics["annualized_return"] = float(
                    (1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / (n_days - 1)) - 1.0
                )
        daily = nav.pct_change()
        metrics["sharpe"] = _annualized_ratio(daily, downside_only=False)
        metrics["sortino"] = _annualized_ratio(daily, downside_only=True)
        metrics["max_drawdown"] = _max_drawdown(ordered)

    if trade_log is None:
        trade_log = pd.DataFrame()
    exits = _exit_rows(trade_log)
    metrics["n_exits"] = int(len(exits))
    if not exits.empty:
        counts = exits["action"].value_counts().to_dict()
        metrics["exit_counts"] = {str(k): int(v) for k, v in counts.items()}
        pnl = _pnl_series(exits)
        if not pnl.empty:
            metrics["n_wins"] = int((pnl > 0).sum())
            metrics["win_rate"] = float(metrics["n_wins"] / len(exits))
            metrics["exit_pnl_net"] = float(pnl.sum())
            grouped = (
                pd.DataFrame({"action": exits["action"].to_numpy(), "pnl": pnl.to_numpy()})
                .groupby("action", dropna=False)["pnl"]
                .sum()
            )
            metrics["exit_pnl"] = {str(k): float(v) for k, v in grouped.items()}
        if "days_open" in exits.columns:
            days = pd.to_numeric(exits["days_open"], errors="coerce").dropna()
            if not days.empty:
                metrics["mean_days_open"] = float(days.mean())
        if "expected_halflife" in exits.columns:
            hl = pd.to_numeric(exits["expected_halflife"], errors="coerce").dropna()
            hl = hl.replace([np.inf, -np.inf], np.nan).dropna()
            if not hl.empty:
                metrics["mean_expected_halflife"] = float(hl.mean())

    return metrics
