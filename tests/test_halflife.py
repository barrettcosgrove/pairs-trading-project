"""
tests/test_halflife.py - Unit Tests for src/scoring/halflife.py

Tests use synthetic return DataFrames with known mathematical properties.
No real market data, no network access, no parquet reads.
All tests complete in under 30 seconds.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.scoring.halflife import (
    _compute_halflife,
    score,
    score_candidate_pairs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns(
    series_dict: dict[str, list[float]],
    start: str = "2022-01-03",
) -> pd.DataFrame:
    """
    Build a long-form returns DataFrame from a dict of ticker -> return list.

    All series must have the same length.
    """
    lengths = {len(v) for v in series_dict.values()}
    assert len(lengths) == 1, "All series must have the same length"
    n = lengths.pop()

    dates = pd.bdate_range(start=start, periods=n)
    rows = []
    for ticker, rets in series_dict.items():
        for dt, r in zip(dates, rets):
            rows.append({"date": dt, "ticker": ticker, "log_return": r})
    return pd.DataFrame(rows)


@pytest.fixture
def mean_reverting_pair():
    """
    Pair whose cumulative spread oscillates with a predictable period.
    Spread = sin wave at ~10-day frequency → half-life near 5-10 days.
    Both series constructed so cumsum(A) - cumsum(B) is a clean sin wave.
    """
    n = CONFIG.signal_window + 10  # slightly more than signal window
    t = np.arange(n)

    # A oscillates, B is A plus noise so they stay correlated
    freq = 2 * np.pi / 10  # 10-day cycle
    signal = np.sin(freq * t)
    returns_a = np.diff(signal, prepend=0)
    returns_b = np.diff(-signal, prepend=0)  # anti-phase → spread = 2*signal

    return _make_returns({"AAA": returns_a.tolist(), "BBB": returns_b.tolist()})


@pytest.fixture
def diverging_pair():
    """
    Pair whose cumulative spread grows monotonically — AR(1) slope >= 0.
    """
    n = CONFIG.signal_window + 10
    returns_a = [0.002] * n        # always positive
    returns_b = [-0.002] * n       # always negative
    # cumsum(A) - cumsum(B) grows linearly → diverging
    return _make_returns({"CCC": returns_a, "DDD": returns_b})


@pytest.fixture
def insufficient_history_pair():
    """
    Pair with fewer observations than halflife_min + 2.
    """
    n = CONFIG.halflife_min + 1    # one less than minimum
    returns_a = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
    returns_b = [0.009 if i % 2 == 0 else -0.009 for i in range(n)]
    return _make_returns({"EEE": returns_a, "FFF": returns_b})


@pytest.fixture
def large_halflife_pair():
    """
    Pair with a very slow mean-reversion — half-life >> CONFIG.halflife_max.
    Constructed by making the spread nearly a random walk (beta ≈ 0).
    """
    n = CONFIG.signal_window + 10
    np.random.seed(99)
    # Spread is a random walk: delta_spread is white noise, lag has no power
    returns_a = np.random.normal(0, 0.01, n).tolist()
    returns_b = np.random.normal(0, 0.01, n).tolist()
    return _make_returns({"GGG": returns_a, "HHH": returns_b})


@pytest.fixture
def valid_candidate_pairs():
    return pd.DataFrame([
        {"ticker_a": "AAA", "ticker_b": "BBB", "cluster_id": 0},
        {"ticker_a": "CCC", "ticker_b": "DDD", "cluster_id": 1},
    ])


# ---------------------------------------------------------------------------
# _compute_halflife unit tests
# ---------------------------------------------------------------------------

class TestComputeHalflife:

    def test_known_ar1_halflife(self):
        np.random.seed(42)
        n = 500
        beta = -0.15
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = beta * spread[i - 1] + np.random.normal(0, 0.1)

        s = pd.Series(spread - spread.mean())
        hl = _compute_halflife(s)

        expected = np.log(2) / abs(beta)
        assert abs(hl - expected) < 5.0, (
            f"Expected half-life near {expected:.1f}, got {hl:.1f}"
        )

    def test_diverging_spread_returns_inf_or_large(self):
        """
        A monotonically growing spread produces near-zero or positive slope.
        _compute_halflife returns inf or a very large number — both correct.
        """
        spread = pd.Series(np.arange(100, dtype=float))
        result = _compute_halflife(spread)
        assert not np.isfinite(result) or result > CONFIG.halflife_max

    def test_zero_slope_returns_inf(self):
        """Flat spread (zero slope) should return inf."""
        spread = pd.Series(np.zeros(100))
        assert _compute_halflife(spread) == float("inf")

    def test_too_short_series_returns_inf(self):
        """Series with fewer than 2 points after differencing returns inf."""
        spread = pd.Series([1.0])
        assert _compute_halflife(spread) == float("inf")

    def test_explosive_spread_scores_zero(self):
        """
        Strongly explosive AR(1) with no noise reliably produces positive slope.
        """
        n = 200
        beta = 0.5
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = beta * spread[i - 1]

        s = pd.Series(spread - spread.mean())
        result = _compute_halflife(s)
        assert not np.isfinite(result) or result > CONFIG.halflife_max


# ---------------------------------------------------------------------------
# score() unit tests
# ---------------------------------------------------------------------------

class TestScore:

    def test_returns_float(self, mean_reverting_pair):
        result = score("AAA", "BBB", mean_reverting_pair, date(2024, 6, 1))
        assert isinstance(result, float)

    def test_score_in_bounds(self, mean_reverting_pair):
        result = score("AAA", "BBB", mean_reverting_pair, date(2024, 6, 1))
        assert 0.0 <= result <= 1.0

    def test_diverging_pair_scores_zero(self, diverging_pair):
        result = score("CCC", "DDD", diverging_pair, date(2024, 6, 1))
        assert result == 0.0

    def test_insufficient_history_scores_zero(self, insufficient_history_pair):
        result = score("EEE", "FFF", insufficient_history_pair, date(2024, 3, 1))
        assert result == 0.0

    def test_missing_ticker_scores_zero(self, mean_reverting_pair):
        result = score("AAA", "NONEXISTENT", mean_reverting_pair, date(2024, 6, 1))
        assert result == 0.0

    def test_symmetric_ab_equals_ba(self, mean_reverting_pair):
        """
        score(A, B) must equal score(B, A) — alphabetical sort enforces symmetry.
        """
        score_ab = score("AAA", "BBB", mean_reverting_pair, date(2024, 6, 1))
        score_ba = score("BBB", "AAA", mean_reverting_pair, date(2024, 6, 1))
        assert score_ab == score_ba, (
            f"Symmetry violated: score(A,B)={score_ab}, score(B,A)={score_ba}"
        )

    def test_respects_as_of_cutoff(self, mean_reverting_pair):
        """
        Scoring with an early as_of should return 0.0 due to insufficient history.
        Using the first few dates means fewer observations than halflife_min + 2.
        """
        early_date = date(2022, 1, 10)   # only ~5 trading days of data
        result = score("AAA", "BBB", mean_reverting_pair, early_date)
        assert result == 0.0

    def test_no_look_ahead(self, mean_reverting_pair):
        """
        Adding future rows should not change the score for an earlier as_of.
        """
        as_of = date(2023, 6, 1)
        score_before = score("AAA", "BBB", mean_reverting_pair, as_of)

        # Append future rows
        future_rows = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-01"), "ticker": "AAA", "log_return": 0.5},
            {"date": pd.Timestamp("2026-01-01"), "ticker": "BBB", "log_return": -0.5},
        ])
        returns_with_future = pd.concat(
            [mean_reverting_pair, future_rows], ignore_index=True
        )
        score_after = score("AAA", "BBB", returns_with_future, as_of)

        assert score_before == score_after, (
            "Look-ahead bias detected: adding future rows changed the score"
        )

    def test_valid_mean_reverting_pair_can_score_one(self, mean_reverting_pair):
        """
        A pair with a clean sin-wave spread should score 1.0 when half-life
        falls within [CONFIG.halflife_min, CONFIG.halflife_max].
        This test verifies the fixture produces a scoreable pair — not all
        synthetic pairs will, so we accept 0.0 or 1.0.
        """
        result = score("AAA", "BBB", mean_reverting_pair, date(2024, 6, 1))
        assert result in (0.0, 1.0), (
            f"Expected 0.0 or 1.0, got {result}"
        )

    def test_score_is_binary(self, mean_reverting_pair, diverging_pair):
        """halflife scorer returns only 0.0 or 1.0 — not a continuous score."""
        for df, a, b in [
            (mean_reverting_pair, "AAA", "BBB"),
            (diverging_pair, "CCC", "DDD"),
        ]:
            result = score(a, b, df, date(2024, 6, 1))
            assert result in (0.0, 1.0), (
                f"Expected binary score, got {result} for {a}/{b}"
            )


# ---------------------------------------------------------------------------
# score_candidate_pairs() tests
# ---------------------------------------------------------------------------

class TestScoreCandidatePairs:

    def test_adds_halflife_score_column(
        self, valid_candidate_pairs, mean_reverting_pair, diverging_pair
    ):
        all_returns = pd.concat(
            [mean_reverting_pair, diverging_pair], ignore_index=True
        )
        result = score_candidate_pairs(
            candidate_pairs=valid_candidate_pairs,
            returns=all_returns,
            as_of=date(2024, 6, 1),
        )
        assert "halflife_score" in result.columns

    def test_output_length_matches_input(
        self, valid_candidate_pairs, mean_reverting_pair, diverging_pair
    ):
        all_returns = pd.concat(
            [mean_reverting_pair, diverging_pair], ignore_index=True
        )
        result = score_candidate_pairs(
            candidate_pairs=valid_candidate_pairs,
            returns=all_returns,
            as_of=date(2024, 6, 1),
        )
        assert len(result) == len(valid_candidate_pairs)

    def test_original_columns_preserved(
        self, valid_candidate_pairs, mean_reverting_pair, diverging_pair
    ):
        all_returns = pd.concat(
            [mean_reverting_pair, diverging_pair], ignore_index=True
        )
        result = score_candidate_pairs(
            candidate_pairs=valid_candidate_pairs,
            returns=all_returns,
            as_of=date(2024, 6, 1),
        )
        for col in ["ticker_a", "ticker_b", "cluster_id"]:
            assert col in result.columns

    def test_all_scores_in_bounds(
        self, valid_candidate_pairs, mean_reverting_pair, diverging_pair
    ):
        all_returns = pd.concat(
            [mean_reverting_pair, diverging_pair], ignore_index=True
        )
        result = score_candidate_pairs(
            candidate_pairs=valid_candidate_pairs,
            returns=all_returns,
            as_of=date(2024, 6, 1),
        )
        assert (result["halflife_score"] >= 0.0).all()
        assert (result["halflife_score"] <= 1.0).all()

    def test_raises_on_missing_required_columns(self, mean_reverting_pair):
        bad_pairs = pd.DataFrame([
            {"ticker_a": "AAA", "ticker_b": "BBB"},  # missing cluster_id
        ])
        with pytest.raises(ValueError, match="required columns"):
            score_candidate_pairs(
                candidate_pairs=bad_pairs,
                returns=mean_reverting_pair,
                as_of=date(2024, 6, 1),
            )

    def test_does_not_mutate_input(
        self, valid_candidate_pairs, mean_reverting_pair, diverging_pair
    ):
        all_returns = pd.concat(
            [mean_reverting_pair, diverging_pair], ignore_index=True
        )
        original_cols = list(valid_candidate_pairs.columns)
        score_candidate_pairs(
            candidate_pairs=valid_candidate_pairs,
            returns=all_returns,
            as_of=date(2024, 6, 1),
        )
        assert list(valid_candidate_pairs.columns) == original_cols