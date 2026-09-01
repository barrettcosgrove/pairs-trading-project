"""
tests/test_metrics.py — Synthetic tests for performance.compute and reporting.

No parquet, no yfinance, no outputs/ from a live backtest.
"""

from datetime import date

import pandas as pd
import pytest

from src.metrics.performance import compute
from src.metrics.reporting import (
    format_metrics_summary,
    generate_report,
    plot_blocked_entries,
    plot_exit_type_mix,
    plot_monthly_returns_heatmap,
    plot_nav_and_drawdown,
)


def _nav_frame(values: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    """Build a dated NAV series from a list of levels."""
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"date": dates, "nav": values})


def test_compute_total_return_and_max_drawdown():
    nav = _nav_frame([100.0, 110.0, 105.0])
    metrics = compute(pd.DataFrame(), nav)
    assert metrics["n_days"] == 3
    assert metrics["start_nav"] == pytest.approx(100.0)
    assert metrics["end_nav"] == pytest.approx(105.0)
    assert metrics["total_return"] == pytest.approx(0.05)
    assert metrics["max_drawdown"] == pytest.approx((110.0 - 105.0) / 110.0)
    assert metrics["sharpe"] == metrics["sharpe"]
    assert metrics["n_exits"] == 0


def test_compute_uses_drawdown_from_peak_column():
    nav = _nav_frame([100.0, 110.0, 105.0])
    nav["drawdown_from_peak"] = [0.0, 0.0, 0.20]
    metrics = compute(pd.DataFrame(), nav)
    assert metrics["max_drawdown"] == pytest.approx(0.20)


def test_compute_exit_mix_ignores_entries():
    trades = pd.DataFrame(
        {
            "date": [date(2022, 1, 3), date(2022, 1, 10), date(2022, 1, 20)],
            "action": ["LONG_SPREAD", "TAKE_PROFIT", "STOP_LOSS"],
            "realized_net_usd": [0.0, 100.0, -40.0],
            "pnl": [0.0, 999.0, -999.0],
            "days_open": [0, 5, 8],
            "expected_halflife": [10.0, 10.0, 12.0],
        }
    )
    metrics = compute(trades, _nav_frame([100.0, 101.0]))
    assert metrics["n_exits"] == 2
    assert metrics["n_wins"] == 1
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["exit_counts"]["TAKE_PROFIT"] == 1
    assert metrics["exit_counts"]["STOP_LOSS"] == 1
    assert "LONG_SPREAD" not in metrics["exit_counts"]
    assert metrics["exit_pnl_net"] == pytest.approx(60.0)
    assert metrics["exit_pnl"]["TAKE_PROFIT"] == pytest.approx(100.0)
    assert metrics["mean_days_open"] == pytest.approx(6.5)
    assert metrics["mean_expected_halflife"] == pytest.approx(11.0)


def test_format_metrics_summary_contains_return():
    metrics = compute(pd.DataFrame(), _nav_frame([100.0, 103.0]))
    text = format_metrics_summary(metrics)
    assert "Total return" in text
    assert "3.00%" in text


def test_plot_nav_and_drawdown_writes_png(tmp_path):
    out = tmp_path / "nav_and_drawdown.png"
    path = plot_nav_and_drawdown(_nav_frame([100.0, 101.0, 99.0]), out)
    assert path == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_nav_skips_empty(tmp_path):
    out = tmp_path / "nav_and_drawdown.png"
    path = plot_nav_and_drawdown(pd.DataFrame(), out)
    assert path is None
    assert not out.exists()


def test_monthly_heatmap_writes_png(tmp_path):
    dates = pd.bdate_range("2022-01-03", "2022-03-31")
    nav = pd.DataFrame(
        {"date": dates, "nav": 100.0 * (1.001 ** pd.Series(range(len(dates))))}
    )
    out = tmp_path / "monthly_returns_heatmap.png"
    path = plot_monthly_returns_heatmap(nav, out)
    assert path == out
    assert out.is_file()


def test_exit_mix_and_blocked_write_pngs(tmp_path):
    trades = pd.DataFrame(
        {
            "action": ["LONG_SPREAD", "TAKE_PROFIT", "STOP_LOSS", "EARNINGS_EXIT"],
            "realized_net_usd": [0.0, 50.0, -20.0, -10.0],
        }
    )
    blocked = pd.DataFrame(
        {"reason": ["vix", "vix", "cooldown", "entry_band"]}
    )
    exit_path = plot_exit_type_mix(trades, tmp_path / "exit_type_mix.png")
    blocked_path = plot_blocked_entries(blocked, tmp_path / "blocked_entries.png")
    assert exit_path is not None and exit_path.is_file()
    assert blocked_path is not None and blocked_path.is_file()


def test_generate_report_skips_empty_charts(tmp_path):
    written = generate_report(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path,
    )
    assert "metrics_summary.txt" in written
    assert (tmp_path / "metrics_summary.txt").is_file()
    assert "nav_and_drawdown.png" not in written
    assert "exit_type_mix.png" not in written
    assert "blocked_entries.png" not in written
