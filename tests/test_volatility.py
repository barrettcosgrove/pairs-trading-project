"""
tests/test_volatility.py - Unit Tests for src/scoring/volatility.py

Tests use synthetic return DataFrames with known mathematical properties.
No real market data, no network access, no parquet reads.
All tests complete in under 30 seconds.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.scoring.volatility import score, score_candidate_pairs


N_OBSERVATIONS = CONFIG.volatility_long_window + CONFIG.volatility_short_window
INSUFFICIENT_OBSERVATIONS = CONFIG.volatility_short_window
START_DATE = "2023-01-02"
FULL_AS_OF = pd.bdate_range(START_DATE, periods=N_OBSERVATIONS)[-1].date()
EARLY_AS_OF = pd.bdate_range(
    START_DATE,
    periods=CONFIG.volatility_short_window // 2,
)[-1].date()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_returns(
    series_dict: dict[str, np.ndarray],
    start: str = START_DATE,
) -> pd.DataFrame:
    """
    Build a long-form returns DataFrame from a dict of ticker -> log_return.
    """
    lengths = {len(v) for v in series_dict.values()}
    assert len(lengths) == 1, "All series must have the same length"
    n = lengths.pop()

    dates = pd.bdate_range(start=start, periods=n)
    rows = []
    for ticker, returns in series_dict.items():
        for dt, log_return in zip(dates, returns):
            rows.append({"date": dt, "ticker": ticker, "log_return": log_return})
    return pd.DataFrame(rows)


@pytest.fixture
def compatible_vol_pair():
    """
    Pair with similar volatility over short and long windows.
    """
    np.random.seed(42)
    returns_a = np.random.normal(0.0, 0.01, N_OBSERVATIONS)
    returns_b = np.random.normal(0.0, 0.011, N_OBSERVATIONS)

    return _make_returns({"AAA": returns_a, "BBB": returns_b})


@pytest.fixture
def incompatible_vol_pair():
    """
    Pair where one leg is much more volatile than the other.
    """
    np.random.seed(7)
    returns_a = np.random.normal(0.0, 0.01, N_OBSERVATIONS)
    returns_b = np.random.normal(0.0, 0.04, N_OBSERVATIONS)

    return _make_returns({"CCC": returns_a, "DDD": returns_b})


@pytest.fixture
def insufficient_history_pair():
    """
    Pair with fewer observations than CONFIG.volatility_long_window.
    """
    np.random.seed(8)
    returns_a = np.random.normal(0.0, 0.01, INSUFFICIENT_OBSERVATIONS)
    returns_b = np.random.normal(0.0, 0.011, INSUFFICIENT_OBSERVATIONS)

    return _make_returns({"EEE": returns_a, "FFF": returns_b})


@pytest.fixture
def identical_vol_pair():
    """
    Pair with exactly matching return volatility.
    """
    np.random.seed(9)
    returns_a = np.random.normal(0.0, 0.01, N_OBSERVATIONS)
    returns_b = returns_a.copy()

    return _make_returns({"GGG": returns_a, "HHH": returns_b})


@pytest.fixture
def combined_returns(compatible_vol_pair, incompatible_vol_pair):
    """
    Combined return fixture for batch candidate-pair scoring.
    """
    return pd.concat(
        [compatible_vol_pair, incompatible_vol_pair],
        ignore_index=True,
    )


@pytest.fixture
def candidate_pairs():
    """
    Candidate pair DataFrame with required scorer columns.
    """
    return pd.DataFrame(
        [
            {"ticker_a": "AAA", "ticker_b": "BBB", "cluster_id": 0},
            {"ticker_a": "CCC", "ticker_b": "DDD", "cluster_id": 0},
        ]
    )


# ---------------------------------------------------------------------------
# score() unit tests
# ---------------------------------------------------------------------------

class TestScore:

    def test_returns_float(self, compatible_vol_pair):
        """score() returns a float."""
        result = score("AAA", "BBB", compatible_vol_pair, FULL_AS_OF)

        assert isinstance(result, float)

    def test_score_in_bounds(self, compatible_vol_pair):
        """score() returns a value in [0.0, 1.0]."""
        result = score("AAA", "BBB", compatible_vol_pair, FULL_AS_OF)

        assert 0.0 <= result <= 1.0

    def test_compatible_pair_scores_above_zero(self, compatible_vol_pair):
        """Compatible volatility pair scores above zero."""
        result = score("AAA", "BBB", compatible_vol_pair, FULL_AS_OF)

        assert result > 0.0

    def test_incompatible_pair_scores_zero(self, incompatible_vol_pair):
        """Short-window volatility ratio above the hard gate scores zero."""
        result = score("CCC", "DDD", incompatible_vol_pair, FULL_AS_OF)

        assert result == 0.0

    def test_insufficient_history_scores_zero(self, insufficient_history_pair):
        """Fewer than CONFIG.volatility_long_window observations scores zero."""
        result = score("EEE", "FFF", insufficient_history_pair, FULL_AS_OF)

        assert result == 0.0

    def test_missing_ticker_scores_zero(self, compatible_vol_pair):
        """Missing ticker input scores zero."""
        result = score("AAA", "ZZZ", compatible_vol_pair, FULL_AS_OF)

        assert result == 0.0

    def test_symmetric_ab_equals_ba(self, compatible_vol_pair):
        """Volatility compatibility score is symmetric for pair order."""
        result_ab = score("AAA", "BBB", compatible_vol_pair, FULL_AS_OF)
        result_ba = score("BBB", "AAA", compatible_vol_pair, FULL_AS_OF)

        assert result_ab == result_ba

    def test_respects_as_of_cutoff(self, compatible_vol_pair):
        """Early as_of dates with too little history score zero."""
        result = score("AAA", "BBB", compatible_vol_pair, EARLY_AS_OF)

        assert result == 0.0

    def test_no_look_ahead(self, compatible_vol_pair):
        """Future rows after as_of do not change the earlier score."""
        as_of = pd.bdate_range(START_DATE, periods=CONFIG.volatility_long_window)[-1].date()
        baseline = score("AAA", "BBB", compatible_vol_pair, as_of)

        np.random.seed(10)
        future_dates = pd.bdate_range(
            pd.Timestamp(FULL_AS_OF) + pd.offsets.BDay(),
            periods=CONFIG.volatility_short_window,
        )
        future_rows = []
        for ticker in ["AAA", "BBB"]:
            future_returns = np.random.normal(0.0, 0.05, len(future_dates))
            for dt, log_return in zip(future_dates, future_returns):
                future_rows.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "log_return": log_return,
                    }
                )

        with_future = pd.concat(
            [compatible_vol_pair, pd.DataFrame(future_rows)],
            ignore_index=True,
        )
        result = score("AAA", "BBB", with_future, as_of)

        assert result == baseline

    def test_higher_compatibility_scores_higher(
        self,
        compatible_vol_pair,
        incompatible_vol_pair,
    ):
        """Compatible pair scores at least as high as incompatible pair."""
        compatible_score = score("AAA", "BBB", compatible_vol_pair, FULL_AS_OF)
        incompatible_score = score("CCC", "DDD", incompatible_vol_pair, FULL_AS_OF)

        assert compatible_score >= incompatible_score

    def test_identical_volatility_scores_high(self, identical_vol_pair):
        """Identical volatility series score close to one."""
        result = score("GGG", "HHH", identical_vol_pair, FULL_AS_OF)

        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_candidate_pairs() unit tests
# ---------------------------------------------------------------------------

class TestScoreCandidatePairs:

    def test_adds_volatility_score_column(self, candidate_pairs, combined_returns):
        """score_candidate_pairs() adds a volatility_score column."""
        result = score_candidate_pairs(candidate_pairs, combined_returns, FULL_AS_OF)

        assert "volatility_score" in result.columns

    def test_output_length_matches_input(self, candidate_pairs, combined_returns):
        """score_candidate_pairs() preserves row count."""
        result = score_candidate_pairs(candidate_pairs, combined_returns, FULL_AS_OF)

        assert len(result) == len(candidate_pairs)

    def test_original_columns_preserved(self, candidate_pairs, combined_returns):
        """score_candidate_pairs() preserves required input columns."""
        result = score_candidate_pairs(candidate_pairs, combined_returns, FULL_AS_OF)

        for column in ["ticker_a", "ticker_b", "cluster_id"]:
            assert column in result.columns

    def test_all_scores_in_bounds(self, candidate_pairs, combined_returns):
        """All batch volatility scores are in [0.0, 1.0]."""
        result = score_candidate_pairs(candidate_pairs, combined_returns, FULL_AS_OF)

        assert result["volatility_score"].between(0.0, 1.0).all()

    def test_raises_on_missing_required_columns(self, combined_returns):
        """Missing candidate-pair columns raise a required columns ValueError."""
        bad_pairs = pd.DataFrame(
            [
                {"ticker_a": "AAA", "ticker_b": "BBB"},
            ]
        )

        with pytest.raises(ValueError, match="required columns"):
            score_candidate_pairs(bad_pairs, combined_returns, FULL_AS_OF)

    def test_does_not_mutate_input(self, candidate_pairs, combined_returns):
        """score_candidate_pairs() does not mutate the input DataFrame columns."""
        original_columns = candidate_pairs.columns.tolist()

        score_candidate_pairs(candidate_pairs, combined_returns, FULL_AS_OF)

        assert candidate_pairs.columns.tolist() == original_columns
