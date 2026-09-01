"""
tests/test_scoring_integration.py - Integration Tests for Scoring Pipeline

Tests use synthetic candidate pairs, returns, and prices. No real market data,
no network access, and no parquet reads.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import src.scoring.cointegration as cointegration
from src.config import CONFIG
from src.scoring.candidate_pairs import build_candidate_pairs
from src.scoring.cointegration import score_candidate_pairs as score_cointegration
from src.scoring.correlation_stability import (
    score_candidate_pairs as score_corr_stability,
)
from src.scoring.halflife import score_candidate_pairs as score_halflife
from src.scoring.volatility import score_candidate_pairs as score_volatility

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
CONSTANT_VOLUME = 1_000_000
BASE_RETURN_VOLATILITY = 0.01
N_OBSERVATIONS = (
    CONFIG.correlation_stability_historical_window + CONFIG.volatility_short_window
)
SCORER_COLUMNS = [
    "correlation_stability_score",
    "halflife_score",
    "cointegration_score",
    "volatility_score",
]
# Halflife batch adds raw half-life (not a [0,1] score); present after score_halflife.
HALFLIFE_VALUE_COLUMN = "halflife_value"


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


@pytest.fixture
def scoring_context():
    """
    Build shared synthetic clusters, returns, prices, as_of, and candidates.
    """
    clusters = {0: ["AAA", "BBB", "CCC"], 1: ["DDD", "EEE"]}
    dates = pd.bdate_range("2023-01-02", periods=N_OBSERVATIONS)

    np.random.seed(0)
    returns_by_ticker = {
        ticker: np.random.normal(0.0, BASE_RETURN_VOLATILITY, N_OBSERVATIONS)
        for ticker in TICKERS
    }

    return_rows = []
    price_rows = []
    for ticker, returns in returns_by_ticker.items():
        prices = 100 * np.exp(np.cumsum(returns))
        for dt, log_return, adj_close in zip(dates, returns, prices):
            return_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "log_return": log_return,
                }
            )
            price_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "adj_close": adj_close,
                    "volume": CONSTANT_VOLUME,
                }
            )

    return {
        "clusters": clusters,
        "returns_df": pd.DataFrame(return_rows),
        "prices_df": pd.DataFrame(price_rows),
        "as_of": dates[-1].date(),
        "candidate_pairs": build_candidate_pairs(clusters),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_scoring_pipeline(
    candidate_pairs: pd.DataFrame,
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of,
) -> pd.DataFrame:
    """
    Run all implemented scorers in pipeline order.
    """
    scored = score_corr_stability(candidate_pairs, returns_df, as_of)
    scored = score_halflife(scored, returns_df, as_of)
    scored = score_cointegration(scored, prices_df, as_of)
    scored = score_volatility(scored, returns_df, as_of)

    if "cointegration_pvalue" in scored.columns:
        scored = scored.drop(columns=["cointegration_pvalue"])
    if "halflife_canonical_order" in scored.columns:
        scored = scored.drop(columns=["halflife_canonical_order"])

    return scored


# ---------------------------------------------------------------------------
# Scoring pipeline integration tests
# ---------------------------------------------------------------------------

class TestScoringPipeline:

    def test_candidate_pairs_has_correct_shape(self, scoring_context):
        """Cluster sizes [3, 2] produce all unique within-cluster pairs."""
        clusters = scoring_context["clusters"]
        candidate_pairs = scoring_context["candidate_pairs"]
        expected_pairs = sum(
            len(list(combinations(tickers, 2))) for tickers in clusters.values()
        )

        assert len(candidate_pairs) == expected_pairs

    def test_all_scorers_add_their_column(self, scoring_context):
        """Sequential scorer pipeline adds every implemented score column."""
        result = _run_scoring_pipeline(
            scoring_context["candidate_pairs"],
            scoring_context["returns_df"],
            scoring_context["prices_df"],
            scoring_context["as_of"],
        )
        expected_columns = [
            "ticker_a",
            "ticker_b",
            "cluster_id",
            *SCORER_COLUMNS,
        ]

        for column in expected_columns:
            assert column in result.columns

    def test_no_pairs_cross_cluster_boundaries(self, scoring_context):
        """Candidate pairs never cross cluster boundaries."""
        clusters = scoring_context["clusters"]
        ticker_to_cluster = {
            ticker: cluster_id
            for cluster_id, tickers in clusters.items()
            for ticker in tickers
        }

        for row in scoring_context["candidate_pairs"].itertuples(index=False):
            assert ticker_to_cluster[row.ticker_a] == ticker_to_cluster[row.ticker_b]

    def test_all_scores_in_bounds_after_pipeline(self, scoring_context):
        """All score columns remain bounded in [0.0, 1.0]."""
        result = _run_scoring_pipeline(
            scoring_context["candidate_pairs"],
            scoring_context["returns_df"],
            scoring_context["prices_df"],
            scoring_context["as_of"],
        )

        for column in SCORER_COLUMNS:
            assert result[column].between(0.0, 1.0).all()

    def test_scorer_pipeline_is_idempotent(self, scoring_context):
        """Running the scorer pipeline twice produces identical scores."""
        first = _run_scoring_pipeline(
            scoring_context["candidate_pairs"],
            scoring_context["returns_df"],
            scoring_context["prices_df"],
            scoring_context["as_of"],
        )
        second = _run_scoring_pipeline(
            scoring_context["candidate_pairs"],
            scoring_context["returns_df"],
            scoring_context["prices_df"],
            scoring_context["as_of"],
        )

        pd.testing.assert_frame_equal(first[SCORER_COLUMNS], second[SCORER_COLUMNS])

    def test_scorer_columns_do_not_overlap(self, scoring_context):
        """Scorers add bounded score columns plus halflife_value metadata."""
        candidate_pairs = scoring_context["candidate_pairs"]
        result = _run_scoring_pipeline(
            candidate_pairs,
            scoring_context["returns_df"],
            scoring_context["prices_df"],
            scoring_context["as_of"],
        )
        added_columns = set(result.columns).difference(candidate_pairs.columns)

        assert added_columns == set(SCORER_COLUMNS) | {HALFLIFE_VALUE_COLUMN}
