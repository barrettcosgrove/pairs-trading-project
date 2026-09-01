"""
tests/test_signals.py - Unit Tests for src/signals/entry_exit.py

Covers entry/exit z-score mapping and the Round 3 fresh-cross entry rule:
a flat pair whose z-score was already beyond the entry band yesterday must
not enter today (prevents entering mid-divergence on the score date).

Synthetic price paths only — no network access, no parquet reads.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.signals.entry_exit import get_signal

# β=1, μ=0 formation stats: spread = log(A) - log(B); with B pinned at 100,
# z = (log(A) - log(100)) / SIGMA. Prices below are chosen per-day to hit
# exact z values.
SIGMA = 0.02


def _price_for_z(z: float) -> float:
    """Return the leg-A price that produces formation z-score ``z``."""
    return float(100.0 * math.exp(z * SIGMA))


def _make_prices(z_path: list[float], start: str = "2022-01-03") -> pd.DataFrame:
    """
    Build a long-form price frame where leg A tracks the given z-score path.

    Leg B stays at 100 so the formation spread moves exactly with leg A.
    """
    dates = pd.bdate_range(start=start, periods=len(z_path))
    rows = []
    for dt, z in zip(dates, z_path):
        rows.append({"date": dt, "ticker": "AAA", "adj_close": _price_for_z(z)})
        rows.append({"date": dt, "ticker": "BBB", "adj_close": 100.0})
    return pd.DataFrame(rows)


def _signal(prices: pd.DataFrame, cfg=None, **kwargs) -> str:
    as_of = prices["date"].max().date()
    defaults = dict(
        beta_formation=1.0,
        mean_formation=0.0,
        std_formation=SIGMA,
        expected_halflife=10.0,
        days_open=0,
        current_position=None,
        config=cfg or CONFIG,
    )
    defaults.update(kwargs)
    return get_signal("AAA", "BBB", prices, as_of, **defaults)


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def test_fresh_cross_short_entry():
    """z moves 1.0 -> beyond entry: SHORT_SPREAD fires on the cross day."""
    z_entry = CONFIG.entry_zscore + 0.3
    prices = _make_prices([0.0, 0.5, 1.0, z_entry])
    assert _signal(prices) == "SHORT_SPREAD"


def test_fresh_cross_long_entry():
    z_entry = -(CONFIG.entry_zscore + 0.3)
    prices = _make_prices([0.0, -0.5, -1.0, z_entry])
    assert _signal(prices) == "LONG_SPREAD"


def test_no_entry_without_cross():
    """z beyond the band yesterday AND today: no entry (mid-divergence)."""
    z_out = CONFIG.entry_zscore + 0.3
    prices = _make_prices([0.0, 1.0, z_out, z_out + 0.2])
    assert _signal(prices) == "HOLD"


def test_entry_without_cross_when_flag_disabled():
    cfg = replace(CONFIG, entry_requires_cross=False)
    z_out = CONFIG.entry_zscore + 0.3
    prices = _make_prices([0.0, 1.0, z_out, z_out + 0.2])
    assert _signal(prices, cfg=cfg) == "SHORT_SPREAD"


def test_no_entry_inside_band():
    prices = _make_prices([0.0, 0.2, -0.3, 0.5])
    assert _signal(prices) == "HOLD"


def test_momentum_breakout_blocks_entry():
    """A >threshold run-up in leg A within momentum_window blocks entry."""
    n = CONFIG.momentum_window + 2
    # Leg A climbs ~25% over the window — well past momentum_threshold (15%) —
    # while z crosses the entry band on the final day only.
    z_path = list(np.linspace(0.0, CONFIG.entry_zscore - 0.1, n)) + [CONFIG.entry_zscore + 11.0]
    prices = _make_prices(z_path)
    assert _signal(prices) == "HOLD"


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

def test_take_profit_short_spread():
    prices = _make_prices([2.0, 1.0, CONFIG.take_profit_zscore - 0.1])
    sig = _signal(prices, current_position="SHORT_SPREAD", days_open=5)
    assert sig == "TAKE_PROFIT"


def test_stop_loss_short_spread():
    prices = _make_prices([2.0, 3.0, CONFIG.stop_loss_zscore + 0.2])
    sig = _signal(prices, current_position="SHORT_SPREAD", days_open=5)
    assert sig == "STOP_LOSS"


def test_stop_loss_long_spread():
    prices = _make_prices([-2.0, -3.0, -(CONFIG.stop_loss_zscore + 0.2)])
    sig = _signal(prices, current_position="LONG_SPREAD", days_open=5)
    assert sig == "STOP_LOSS"


def test_time_stop_fires_at_limit():
    prices = _make_prices([2.0, 1.9, 1.8])
    sig = _signal(
        prices, current_position="SHORT_SPREAD", days_open=CONFIG.time_stop_days
    )
    assert sig == "TIME_STOP"


def test_hold_between_thresholds_when_open():
    z_mid = (CONFIG.take_profit_zscore + CONFIG.stop_loss_zscore) / 2
    prices = _make_prices([z_mid + 0.5, z_mid + 0.2, z_mid])
    sig = _signal(prices, current_position="SHORT_SPREAD", days_open=5)
    assert sig == "HOLD"


def test_nan_beta_returns_hold():
    prices = _make_prices([0.0, 2.0])
    assert _signal(prices, beta_formation=float("nan")) == "HOLD"


# ---------------------------------------------------------------------------
# Round 4: entry band cap and plateau stop
# ---------------------------------------------------------------------------

def test_entry_band_cap_blocks_gap_cross():
    # Fresh cross, but z gapped straight past entry_zscore_max — no entry.
    prices = _make_prices([0.5, CONFIG.entry_zscore_max + 0.5])
    assert _signal(prices) == "HOLD"


def test_entry_band_cap_allows_normal_cross():
    z_in_band = (CONFIG.entry_zscore + CONFIG.entry_zscore_max) / 2
    prices = _make_prices([0.5, z_in_band])
    assert _signal(prices) == "SHORT_SPREAD"


def test_entry_band_cap_disabled_allows_gap_cross():
    cfg = replace(CONFIG, entry_zscore_max=None)
    prices = _make_prices([0.5, 2.5])
    assert _signal(prices, cfg=cfg) == "SHORT_SPREAD"


def test_plateau_stop_fires_after_consecutive_adverse_days():
    plateau = CONFIG.stop_plateau_zscore + 0.05
    prices = _make_prices([1.5] + [plateau] * CONFIG.stop_plateau_days)
    sig = _signal(
        prices,
        current_position="SHORT_SPREAD",
        days_open=CONFIG.stop_plateau_days,
    )
    assert sig == "PLATEAU_STOP"


def test_plateau_stop_needs_full_consecutive_run():
    plateau = CONFIG.stop_plateau_zscore + 0.05
    # A dip back inside the plateau band on the middle day breaks the run.
    path = [1.5] + [plateau] * (CONFIG.stop_plateau_days - 1) + [1.5, plateau]
    prices = _make_prices(path)
    sig = _signal(
        prices,
        current_position="SHORT_SPREAD",
        days_open=len(path) - 1,
    )
    assert sig == "HOLD"


def test_plateau_stop_long_spread_adverse_is_negative_z():
    plateau = -(CONFIG.stop_plateau_zscore + 0.05)
    prices = _make_prices([-1.5] + [plateau] * CONFIG.stop_plateau_days)
    sig = _signal(
        prices,
        current_position="LONG_SPREAD",
        days_open=CONFIG.stop_plateau_days,
    )
    assert sig == "PLATEAU_STOP"


def test_plateau_stop_disabled():
    cfg = replace(CONFIG, stop_plateau_days=0)
    plateau = CONFIG.stop_plateau_zscore + 0.05
    prices = _make_prices([1.5] + [plateau] * 4)
    sig = _signal(prices, cfg=cfg, current_position="SHORT_SPREAD", days_open=4)
    assert sig == "HOLD"
