"""
tests/test_cointegration.py - Unit Tests for src/scoring/cointegration.py

Tests use synthetic price DataFrames with known mathematical properties.
No real market data, no network access, no parquet reads.
All tests complete in under 30 seconds.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.scoring.cointegration as cointegration
from src.config import CONFIG
from src.scoring.cointegration import score, score_candidate_pairs

N_OBSERVATIONS = CONFIG.signal_window * 5
INSUFFICIENT_OBSERVATIONS = 20
CONSTANT_VOLUME = 1_000_000
START_DATE = "2023-01-02"
FULL_AS_OF = date(2024, 2, 23)
EARLY_AS_OF = date(2023, 1, 27)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fast_cointegration_test(monkeypatch):
    """
    Replace statsmodels coint_johansen with a deterministic fast test double.
    """
    class FakeJohansenResult:
        def __init__(self):
            # Trace stat much larger than any critical value → p-value near 0.001
            self.lr1 = np.array([50.0])
            # Critical values at 90%, 95%, 99%
            self.cvt = np.array([[13.4, 15.5, 19.9]])

    def fake_coint_johansen(df, det_order, k_ar_diff):
        return FakeJohansenResult()

    monkeypatch.setattr(cointegration, "coint_johansen", fake_coint_johansen)


def _make_prices(
    series_dict: dict[str, np.ndarray],
    start: str = START_DATE,
) -> pd.DataFrame:
    """
    Build a long-form prices DataFrame from a dict of ticker -> adj_close.
    """
    lengths = {len(v) for v in series_dict.values()}
    assert len(lengths) == 1, "All series must have the same length"
    n = lengths.pop()

    dates = pd.bdate_range(start=start, periods=n)
    rows = []
    for ticker, prices in series_dict.items():
        for dt, adj_close in zip(dates, prices):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "adj_close": adj_close,
                    "volume": CONSTANT_VOLUME,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def cointegrated_pair_prices():
    """
    Pair whose log prices share a common stochastic trend.
    """
    np.random.seed(42)
    common_trend = np.cumsum(np.random.normal(0.0003, 0.01, N_OBSERVATIONS))
    small_independent_noise = np.random.normal(0.0, 0.01, N_OBSERVATIONS)

    price_a = np.exp(common_trend)
    price_b = 0.8 * price_a * np.exp(small_independent_noise)

    return _make_prices({"AAA": price_a, "BBB": price_b})


@pytest.fixture
def independent_pair_prices():
    """
    Pair whose log prices are independent random walks.
    """
    np.random.seed(43)
    returns_a = np.random.normal(0.0003, 0.02, N_OBSERVATIONS)
    np.random.seed(44)
    returns_b = np.random.normal(0.0003, 0.02, N_OBSERVATIONS)

    price_a = np.exp(np.cumsum(returns_a))
    price_b = np.exp(np.cumsum(returns_b))

    return _make_prices({"CCC": price_a, "DDD": price_b})


@pytest.fixture
def insufficient_history_prices():
    """
    Pair with fewer than the scorer's minimum aligned observations.
    """
    np.random.seed(45)
    common_trend = np.cumsum(np.random.normal(0.0003, 0.01, INSUFFICIENT_OBSERVATIONS))
    small_independent_noise = np.random.normal(0.0, 0.01, INSUFFICIENT_OBSERVATIONS)

    price_a = np.exp(common_trend)
    price_b = 0.8 * price_a * np.exp(small_independent_noise)

    return _make_prices({"EEE": price_a, "FFF": price_b})


@pytest.fixture
def combined_prices(cointegrated_pair_prices, independent_pair_prices):
    """
    Combined price fixture for batch candidate-pair scoring.
    """
    return pd.concat(
        [cointegrated_pair_prices, independent_pair_prices],
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

    def test_returns_float(self, cointegrated_pair_prices):
        """score() returns a float."""
        result = score("AAA", "BBB", cointegrated_pair_prices, FULL_AS_OF)

        assert isinstance(result, float)

    def test_score_in_bounds(self, cointegrated_pair_prices):
        """score() returns a value in [0.0, 1.0]."""
        result = score("AAA", "BBB", cointegrated_pair_prices, FULL_AS_OF)

        assert 0.0 <= result <= 1.0

    def test_insufficient_history_scores_zero(self, insufficient_history_prices):
        """Fewer than the minimum aligned observations scores zero."""
        result = score("EEE", "FFF", insufficient_history_prices, FULL_AS_OF)

        assert result == 0.0

    def test_missing_ticker_scores_zero(self, cointegrated_pair_prices):
        """Missing ticker input scores zero."""
        result = score("AAA", "ZZZ", cointegrated_pair_prices, FULL_AS_OF)

        assert result == 0.0

    def test_symmetric_ab_equals_ba(self, cointegrated_pair_prices):
        """Cointegration score is symmetric for pair order."""
        result_ab = score("AAA", "BBB", cointegrated_pair_prices, FULL_AS_OF)
        result_ba = score("BBB", "AAA", cointegrated_pair_prices, FULL_AS_OF)

        assert result_ab == result_ba

    def test_respects_as_of_cutoff(self, cointegrated_pair_prices):
        """Early as_of dates with too little history score zero."""
        result = score("AAA", "BBB", cointegrated_pair_prices, EARLY_AS_OF)

        assert result == 0.0

    def test_no_look_ahead(self, cointegrated_pair_prices):
        """Future rows after as_of do not change the earlier score."""
        as_of = date(2023, 10, 6)
        baseline = score("AAA", "BBB", cointegrated_pair_prices, as_of)

        np.random.seed(46)
        future_dates = pd.bdate_range("2024-03-01", periods=CONFIG.signal_window)
        future_rows = []
        for ticker in ["AAA", "BBB"]:
            future_prices = np.exp(np.cumsum(np.random.normal(0.0, 0.05, len(future_dates))))
            for dt, adj_close in zip(future_dates, future_prices):
                future_rows.append(
                    {
                        "date": dt,
                        "ticker": ticker,
                        "adj_close": adj_close,
                        "volume": CONSTANT_VOLUME,
                    }
                )

        with_future = pd.concat(
            [cointegrated_pair_prices, pd.DataFrame(future_rows)],
            ignore_index=True,
        )
        result = score("AAA", "BBB", with_future, as_of)

        assert result == baseline

    def test_independent_pair_scores_lower_than_cointegrated(
        self,
        cointegrated_pair_prices,
        independent_pair_prices,
    ):
        """Cointegrated synthetic pair scores at least as high as independent pair."""
        cointegrated_score = score("AAA", "BBB", cointegrated_pair_prices, FULL_AS_OF)
        independent_score = score("CCC", "DDD", independent_pair_prices, FULL_AS_OF)

        assert cointegrated_score >= independent_score


# ---------------------------------------------------------------------------
# score_candidate_pairs() unit tests
# ---------------------------------------------------------------------------

class TestScoreCandidatePairs:

    def test_adds_cointegration_score_column(self, candidate_pairs, combined_prices):
        """score_candidate_pairs() adds a cointegration_score column."""
        result = score_candidate_pairs(candidate_pairs, combined_prices, FULL_AS_OF)

        assert "cointegration_score" in result.columns

    def test_output_length_matches_input(self, candidate_pairs, combined_prices):
        """score_candidate_pairs() preserves row count."""
        result = score_candidate_pairs(candidate_pairs, combined_prices, FULL_AS_OF)

        assert len(result) == len(candidate_pairs)

    def test_original_columns_preserved(self, candidate_pairs, combined_prices):
        """score_candidate_pairs() preserves required input columns."""
        result = score_candidate_pairs(candidate_pairs, combined_prices, FULL_AS_OF)

        for column in ["ticker_a", "ticker_b", "cluster_id"]:
            assert column in result.columns

    def test_all_scores_in_bounds(self, candidate_pairs, combined_prices):
        """All batch cointegration scores are in [0.0, 1.0]."""
        result = score_candidate_pairs(candidate_pairs, combined_prices, FULL_AS_OF)

        assert result["cointegration_score"].between(0.0, 1.0).all()

    def test_raises_on_missing_required_columns(self, combined_prices):
        """Missing candidate-pair columns raise a required columns ValueError."""
        bad_pairs = pd.DataFrame(
            [
                {"ticker_a": "AAA", "ticker_b": "BBB"},
            ]
        )

        with pytest.raises(ValueError, match="required columns"):
            score_candidate_pairs(bad_pairs, combined_prices, FULL_AS_OF)

    def test_does_not_mutate_input(self, candidate_pairs, combined_prices):
        """score_candidate_pairs() does not mutate the input DataFrame columns."""
        original_columns = candidate_pairs.columns.tolist()

        score_candidate_pairs(candidate_pairs, combined_prices, FULL_AS_OF)

        assert candidate_pairs.columns.tolist() == original_columns
