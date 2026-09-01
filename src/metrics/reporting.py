"""
src/metrics/reporting.py — Charts and summary text for script 04

Reads in-memory frames from the backtest CSVs and writes PNGs plus a text
summary under ``outputs/report/``. Missing or empty inputs skip the affected
chart and log a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.metrics.performance import _exit_rows, _pnl_series, compute

logger = logging.getLogger(__name__)

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _ensure_dir(out_dir: Path) -> Path:
    """
    Create ``out_dir`` if needed and return it.

    Args:
        out_dir: Destination directory for report artifacts.

    Returns:
        The same path, after ``mkdir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _prepare_nav(nav_series: pd.DataFrame) -> pd.DataFrame:
    """
    Return NAV rows sorted by date with a datetime index.

    Args:
        nav_series: Raw NAV frame from the engine CSV.

    Returns:
        Copy with ``date`` as index, or an empty frame if unusable.
    """
    if nav_series is None or nav_series.empty or "nav" not in nav_series.columns:
        return pd.DataFrame()
    df = nav_series.copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date").set_index("date")
    return df


def format_metrics_summary(metrics: dict) -> str:
    """
    Render ``compute()`` output as a plain-text block.

    Args:
        metrics: Dict returned by ``src.metrics.performance.compute``.

    Returns:
        Multi-line summary string.
    """
    def _pct(value: float) -> str:
        if value != value:
            return "n/a"
        return f"{value * 100:.2f}%"

    def _num(value: float, digits: int = 2) -> str:
        if value != value:
            return "n/a"
        return f"{value:.{digits}f}"

    def _usd(value: float) -> str:
        if value != value:
            return "n/a"
        return f"${value:,.2f}"

    lines = [
        "ARQ pairs trading — performance summary",
        f"  Trading days       : {metrics.get('n_days', 0)}",
        f"  Start NAV          : {_usd(metrics.get('start_nav', float('nan')))}",
        f"  End NAV            : {_usd(metrics.get('end_nav', float('nan')))}",
        f"  Total return       : {_pct(metrics.get('total_return', float('nan')))}",
        f"  Annualized return  : {_pct(metrics.get('annualized_return', float('nan')))}",
        f"  Sharpe (rf=0)      : {_num(metrics.get('sharpe', float('nan')), 3)}",
        f"  Sortino (rf=0)     : {_num(metrics.get('sortino', float('nan')), 3)}",
        f"  Max drawdown       : {_pct(metrics.get('max_drawdown', float('nan')))}",
        f"  Round-trip exits   : {metrics.get('n_exits', 0)}",
        f"  Wins               : {metrics.get('n_wins', 0)}",
        f"  Win rate           : {_pct(metrics.get('win_rate', float('nan')))}",
        f"  Exit P&L (net)     : {_num(metrics.get('exit_pnl_net', float('nan')))}",
        f"  Mean days open     : {_num(metrics.get('mean_days_open', float('nan')))}",
        f"  Mean expected HL   : {_num(metrics.get('mean_expected_halflife', float('nan')))}",
        "  Exit mix:",
    ]
    counts = metrics.get("exit_counts") or {}
    pnl = metrics.get("exit_pnl") or {}
    if not counts:
        lines.append("    (none)")
    else:
        for action in sorted(counts):
            action_pnl = pnl.get(action, float("nan"))
            pnl_str = f"${action_pnl:,.0f}" if action_pnl == action_pnl else "n/a"
            lines.append(f"    {action:16s} n={counts[action]:3d}  pnl={pnl_str}")
    return "\n".join(lines) + "\n"


def write_metrics_summary(metrics: dict, out_path: Path) -> Path | None:
    """
    Write the text metrics summary to ``out_path``.

    Args:
        metrics: Dict returned by ``compute``.
        out_path: Destination ``.txt`` path.

    Returns:
        ``out_path`` after writing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = format_metrics_summary(metrics)
    out_path.write_text(text)
    return out_path


def plot_nav_and_drawdown(nav_series: pd.DataFrame, out_path: Path) -> Path | None:
    """
    Write a two-panel NAV and drawdown chart.

    Args:
        nav_series: Daily NAV frame with ``date`` and ``nav``.
        out_path: Destination PNG path.

    Returns:
        ``out_path`` if the chart was written, otherwise None.
    """
    df = _prepare_nav(nav_series)
    if df.empty:
        logger.warning("Skipping nav_and_drawdown.png — NAV series is empty.")
        return None

    if "drawdown_from_peak" in df.columns:
        drawdown_pct = pd.to_numeric(df["drawdown_from_peak"], errors="coerce").fillna(0.0) * 100.0
    else:
        peak = df["nav"].cummax()
        drawdown_pct = (df["nav"] / peak - 1.0) * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Portfolio NAV and drawdown", fontsize=14, fontweight="bold")

    ax1.plot(df.index, df["nav"], color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel("NAV ($)")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2.fill_between(df.index, 0, drawdown_pct, color="#d62728", alpha=0.3)
    ax2.plot(df.index, drawdown_pct, color="#d62728", linewidth=1.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monthly_returns_heatmap(nav_series: pd.DataFrame, out_path: Path) -> Path | None:
    """
    Write a year-by-month heatmap of NAV returns.

    Args:
        nav_series: Daily NAV frame with ``date`` and ``nav``.
        out_path: Destination PNG path.

    Returns:
        ``out_path`` if the chart was written, otherwise None.
    """
    df = _prepare_nav(nav_series)
    if df.empty or len(df) < 2:
        logger.warning("Skipping monthly_returns_heatmap.png — NAV series is empty.")
        return None

    monthly = df["nav"].resample("ME").last().pct_change()
    monthly = monthly.dropna()
    if monthly.empty:
        logger.warning("Skipping monthly_returns_heatmap.png — not enough months.")
        return None

    grid = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.to_numpy() * 100.0,
        }
    )
    pivot = grid.pivot(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.6 * len(pivot) + 1.5)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "Monthly return (%)"},
        xticklabels=MONTH_LABELS,
    )
    ax.set_title("Monthly NAV returns (%)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_exit_type_mix(trade_log: pd.DataFrame, out_path: Path) -> Path | None:
    """
    Write a bar chart of exit counts with net P&L labels.

    Args:
        trade_log: Engine trade log.
        out_path: Destination PNG path.

    Returns:
        ``out_path`` if the chart was written, otherwise None.
    """
    exits = _exit_rows(trade_log)
    if exits.empty:
        logger.warning("Skipping exit_type_mix.png — no exit rows.")
        return None

    counts = exits["action"].value_counts()
    pnl = _pnl_series(exits)
    if pnl.empty:
        pnl_by_action = pd.Series(0.0, index=counts.index)
    else:
        pnl_by_action = (
            pd.DataFrame({"action": exits["action"].to_numpy(), "pnl": pnl.to_numpy()})
            .groupby("action")["pnl"]
            .sum()
            .reindex(counts.index)
            .fillna(0.0)
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in pnl_by_action.to_numpy()]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors)
    ax.set_ylabel("Exits")
    ax.set_title("Exit-type mix")
    ax.grid(True, axis="y", alpha=0.3)
    for i, action in enumerate(counts.index):
        label = f"n={int(counts.iloc[i])}\n${pnl_by_action.iloc[i]:,.0f}"
        ax.text(i, counts.iloc[i], label, ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_blocked_entries(blocked_entries: pd.DataFrame, out_path: Path) -> Path | None:
    """
    Write a horizontal bar chart of blocked-entry reasons.

    Args:
        blocked_entries: Engine blocked-entry log with a ``reason`` column.
        out_path: Destination PNG path.

    Returns:
        ``out_path`` if the chart was written, otherwise None.
    """
    if blocked_entries is None or blocked_entries.empty or "reason" not in blocked_entries.columns:
        logger.warning("Skipping blocked_entries.png — no blocked-entry rows.")
        return None

    counts = blocked_entries["reason"].value_counts().sort_values(ascending=True)
    if counts.empty:
        logger.warning("Skipping blocked_entries.png — no blocked-entry rows.")
        return None

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.4 * len(counts) + 1.5)))
    ax.barh(counts.index.astype(str), counts.to_numpy(), color="#1f77b4")
    ax.set_xlabel("Blocked pair-days")
    ax.set_title("Blocked-entry reasons")
    ax.grid(True, axis="x", alpha=0.3)
    for i, value in enumerate(counts.to_numpy()):
        ax.text(value, i, f" {int(value)}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_report(
    trade_log: pd.DataFrame,
    nav_series: pd.DataFrame,
    blocked_entries: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """
    Write the core chart pack and metrics summary into ``out_dir``.

    Empty inputs skip the corresponding chart and log a warning.

    Args:
        trade_log: Engine trade log.
        nav_series: Daily NAV series.
        blocked_entries: Blocked-entry log.
        out_dir: Report output directory.

    Returns:
        Mapping of artifact name to path for files that were written.
    """
    out_dir = _ensure_dir(out_dir)
    written: dict[str, Path] = {}

    metrics = compute(trade_log, nav_series)
    summary_path = write_metrics_summary(metrics, out_dir / "metrics_summary.txt")
    if summary_path is not None:
        written["metrics_summary.txt"] = summary_path

    path = plot_nav_and_drawdown(nav_series, out_dir / "nav_and_drawdown.png")
    if path is not None:
        written["nav_and_drawdown.png"] = path

    path = plot_monthly_returns_heatmap(nav_series, out_dir / "monthly_returns_heatmap.png")
    if path is not None:
        written["monthly_returns_heatmap.png"] = path

    path = plot_exit_type_mix(trade_log, out_dir / "exit_type_mix.png")
    if path is not None:
        written["exit_type_mix.png"] = path

    path = plot_blocked_entries(blocked_entries, out_dir / "blocked_entries.png")
    if path is not None:
        written["blocked_entries.png"] = path

    return written
