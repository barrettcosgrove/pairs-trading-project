"""
tests/test_backtest.py - Unit Tests for src/backtest/portfolio.py drawdown
controls and the engine's drawdown trim.

Covers the Round 3 fixes:
- Hard halt releases after the trim completes plus consecutive non-losing
  days (the old rule required NAV within 5% of the all-time peak with no
  entries allowed — an unreachable deadlock).
- Trim reduces positions to drawdown_trim_factor of pre-halt size after
  drawdown_trim_days, not drawdown_trim_factor ** drawdown_trim_days.

Synthetic data only — no network access, no parquet reads.
"""

from datetime import date

import pandas as pd

from src.backtest.engine import _execute_trim
from src.backtest.portfolio import Portfolio
from src.config import CONFIG


def _halt_portfolio(cfg) -> Portfolio:
    """Drive a portfolio into hard halt with a 12% cash-only drawdown."""
    portfolio = Portfolio(config=cfg)
    portfolio.peak_nav = 100_000.0
    nav = 88_000.0
    entries_ok, size_factor = portfolio.check_drawdown_controls(nav)
    assert entries_ok is False and size_factor == 0.0
    assert portfolio._halted is True
    return portfolio


def test_hard_halt_blocks_entries_at_threshold():
    portfolio = _halt_portfolio(CONFIG)
    assert portfolio._trim_days_remaining == CONFIG.drawdown_trim_days


def test_halt_releases_after_trim_and_recovery_days():
    portfolio = _halt_portfolio(CONFIG)

    # Trim window: engine trims then decrements; entries stay blocked.
    for _ in range(CONFIG.drawdown_trim_days):
        entries_ok, _ = portfolio.check_drawdown_controls(88_000.0)
        assert entries_ok is False
        portfolio.decrement_trim_day()

    # First post-trim call resets the peak to current NAV and starts counting.
    entries_ok, _ = portfolio.check_drawdown_controls(88_000.0)
    assert entries_ok is False
    assert portfolio.peak_nav == 88_000.0

    # Consecutive non-losing (flat counts) days release the halt.
    for day in range(CONFIG.drawdown_recovery_days):
        entries_ok, size_factor = portfolio.check_drawdown_controls(88_000.0)
    assert entries_ok is True
    assert portfolio._halted is False


def test_losing_day_resets_recovery_counter():
    portfolio = _halt_portfolio(CONFIG)
    for _ in range(CONFIG.drawdown_trim_days):
        portfolio.check_drawdown_controls(88_000.0)
        portfolio.decrement_trim_day()
    portfolio.check_drawdown_controls(88_000.0)  # peak reset

    # Almost recovered, then one down day resets the counter.
    for _ in range(CONFIG.drawdown_recovery_days - 1):
        portfolio.check_drawdown_controls(88_000.0)
    entries_ok, _ = portfolio.check_drawdown_controls(87_000.0)
    assert entries_ok is False

    # Needs the full run of non-losing days again.
    for _ in range(CONFIG.drawdown_recovery_days - 1):
        entries_ok, _ = portfolio.check_drawdown_controls(87_000.0)
        assert entries_ok is False
    entries_ok, _ = portfolio.check_drawdown_controls(87_000.0)
    assert entries_ok is True


def test_halt_can_retrigger_after_release():
    portfolio = _halt_portfolio(CONFIG)
    for _ in range(CONFIG.drawdown_trim_days):
        portfolio.check_drawdown_controls(88_000.0)
        portfolio.decrement_trim_day()
    portfolio.check_drawdown_controls(88_000.0)
    for _ in range(CONFIG.drawdown_recovery_days):
        entries_ok, _ = portfolio.check_drawdown_controls(88_000.0)
    assert entries_ok is True

    # A fresh 10% drop from the reset peak (88k) halts again.
    entries_ok, _ = portfolio.check_drawdown_controls(79_000.0)
    assert entries_ok is False
    assert portfolio._halted is True


def test_trim_reaches_target_factor_not_compounded():
    cfg = CONFIG
    portfolio = Portfolio(config=cfg)
    entry_date = date(2022, 1, 3)
    portfolio.open_position(
        ticker_a="AAA",
        ticker_b="BBB",
        cluster_id=0,
        direction="LONG_SPREAD",
        beta=1.0,
        mean=0.0,
        std=0.05,
        expected_halflife=10.0,
        price_a=100.0,
        price_b=100.0,
        dollar_allocation=10_000.0,
        entry_date=entry_date,
    )
    pos = portfolio.positions[("AAA", "BBB")]
    shares_a_start = pos.shares_a

    prices = pd.DataFrame({
        "date": [pd.Timestamp("2022-03-01")] * 2,
        "ticker": ["AAA", "BBB"],
        "adj_close": [100.0, 100.0],
    })

    for _ in range(cfg.drawdown_trim_days):
        _execute_trim(portfolio, prices, date(2022, 3, 1), cfg, [])

    remaining_fraction = pos.shares_a / shares_a_start
    assert abs(remaining_fraction - cfg.drawdown_trim_factor) < 1e-9, (
        f"Trim should leave {cfg.drawdown_trim_factor:.0%} of the position, "
        f"left {remaining_fraction:.2%}"
    )


def test_position_size_respects_max_weight_per_pair():
    portfolio = Portfolio(config=CONFIG)
    nav = 100_000.0
    alloc = portfolio.position_size(nav, size_factor=1.0, active_pairs_count=1)
    assert alloc <= nav * CONFIG.max_weight_per_pair + 1e-9
