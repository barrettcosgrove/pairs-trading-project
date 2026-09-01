"""
tests/test_composite.py — Unit Tests for src/scoring/composite.py

Tests use synthetic return and price DataFrames built entirely in fixtures.
No real market data, no network access, no parquet reads.
All tests complete in under 30 seconds.
"""


import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.config import CONFIG
from src.scoring.candidate_pairs import build_candidate_pairs
from src.scoring.composite import score_candidates

# Columns returned by score_candidates (including formation stats for the engine).
SCORE_CANDIDATES_OUTPUT_COLUMNS = [
    "ticker_a",
    "ticker_b",
    "cluster_id",
    "composite_score",
    "beta_formation",
    "mean_formation",
    "std_formation",
    "halflife_value",
]


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """
    Build 5-ticker, 300-day synthetic returns, prices, clusters, and as_of date.

    Returns a dict with keys: returns, prices, clusters, as_of.
    Cluster 0 has 3 tickers (AAA, BBB, CCC) → 3 candidate pairs.
    Cluster 1 has 2 tickers (DDD, EEE)       → 1 candidate pair.
    """
    np.random.seed(42)

    dates = pd.bdate_range("2022-01-03", periods=300)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    returns_rows = []
    prices_rows = []

    for ticker in tickers:
        rets = np.random.normal(0.0003, 0.015, len(dates))
        price = 100.0
        for dt, r in zip(dates, rets):
            returns_rows.append({"date": dt, "ticker": ticker, "log_return": r})
            price = price * np.exp(r)
            prices_rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )

    returns_df = pd.DataFrame(returns_rows)
    prices_df = pd.DataFrame(prices_rows)

    clusters = {0: ["AAA", "BBB", "CCC"], 1: ["DDD", "EEE"]}
    as_of = dates[-1].date()

    return {
        "returns": returns_df,
        "prices": prices_df,
        "clusters": clusters,
        "as_of": as_of,
    }


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestScoreCandidatesOutputContract:

    def test_returns_dataframe(self, synthetic_data):
        """score_candidates() returns a pd.DataFrame."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_exact(self, synthetic_data):
        """Output columns match the engine contract including formation stats."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS

    def test_composite_score_in_bounds(self, synthetic_data):
        """All composite_score values are in [0.0, 1.0]."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        assert (result["composite_score"] >= 0.0).all()
        assert (result["composite_score"] <= 1.0).all()

    def test_no_cross_cluster_pairs(self, synthetic_data):
        """Every output row has ticker_a and ticker_b belonging to the same cluster."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        ticker_to_cluster = {
            ticker: cid
            for cid, tickers in d["clusters"].items()
            for ticker in tickers
        }
        for _, row in result.iterrows():
            assert ticker_to_cluster[row["ticker_a"]] == ticker_to_cluster[row["ticker_b"]], (
                f"Cross-cluster pair: {row['ticker_a']} (cluster "
                f"{ticker_to_cluster[row['ticker_a']]}) paired with "
                f"{row['ticker_b']} (cluster {ticker_to_cluster[row['ticker_b']]})"
            )

    def test_finalists_per_cluster_respected(self, synthetic_data):
        """No cluster contributes more than CONFIG.finalists_per_cluster rows."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        cluster_counts = result.groupby("cluster_id").size()
        assert (cluster_counts <= CONFIG.finalists_per_cluster).all(), (
            f"Finalist cap ({CONFIG.finalists_per_cluster}) exceeded: "
            f"{cluster_counts.to_dict()}"
        )

    def test_sorted_descending_within_cluster(self, synthetic_data):
        """Within each cluster_id, composite_score is non-increasing."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        for cluster_id, group in result.groupby("cluster_id"):
            scores = group["composite_score"].tolist()
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Cluster {cluster_id}: score at position {i} ({scores[i]:.4f}) "
                    f"is less than score at position {i + 1} ({scores[i + 1]:.4f})"
                )

    def test_index_is_range_index(self, synthetic_data):
        """Output index is a RangeIndex starting at 0."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result.index, pd.RangeIndex)
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestScoreCandidatesEdgeCases:

    def test_empty_clusters_returns_empty_dataframe(self, synthetic_data):
        """score_candidates with empty clusters dict returns empty DataFrame with correct columns."""
        d = synthetic_data
        result = score_candidates(
            clusters={},
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS

    def test_none_clusters_returns_empty_dataframe(self, synthetic_data):
        """score_candidates with None clusters returns empty DataFrame or raises ValueError."""
        d = synthetic_data
        try:
            result = score_candidates(
                clusters=None,
                returns=d["returns"],
                prices=d["prices"],
                as_of=d["as_of"],
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS
        except ValueError:
            pass  # ValueError with a clear message is also acceptable

    def test_single_ticker_cluster_returns_no_pairs(self, synthetic_data):
        """A cluster with one ticker cannot form a pair and produces no output rows."""
        d = synthetic_data
        result = score_candidates(
            clusters={0: ["AAA"]},
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS

    def test_two_ticker_cluster_returns_at_most_one_pair(self, synthetic_data):
        """A two-ticker cluster yields at most one surviving pair in the output."""
        d = synthetic_data
        result = score_candidates(
            clusters={0: ["AAA", "BBB"]},
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 1
        assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS

    def test_all_clusters_singleton_returns_empty(self, synthetic_data):
        """All-singleton clusters produce no pairs and return an empty DataFrame."""
        d = synthetic_data
        result = score_candidates(
            clusters={0: ["AAA"], 1: ["BBB"], 2: ["CCC"]},
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == SCORE_CANDIDATES_OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Canonical ordering
# ---------------------------------------------------------------------------


class TestScoreCandidatesCanonicalOrdering:

    def test_ticker_a_and_ticker_b_are_valid_tickers(self, synthetic_data):
        """All ticker_a and ticker_b values appear in the original cluster ticker lists."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        all_tickers = {t for tickers in d["clusters"].values() for t in tickers}
        assert set(result["ticker_a"]).issubset(all_tickers), (
            f"Unexpected ticker_a values: {set(result['ticker_a']) - all_tickers}"
        )
        assert set(result["ticker_b"]).issubset(all_tickers), (
            f"Unexpected ticker_b values: {set(result['ticker_b']) - all_tickers}"
        )

    def test_no_self_pairs(self, synthetic_data):
        """No output row has ticker_a equal to ticker_b."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        assert (result["ticker_a"] != result["ticker_b"]).all()

    def test_no_duplicate_pairs(self, synthetic_data):
        """No unordered ticker pair appears more than once — canonical ordering enforces one direction."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        pair_sets = [
            frozenset([row["ticker_a"], row["ticker_b"]])
            for _, row in result.iterrows()
        ]
        assert len(pair_sets) == len(set(pair_sets)), (
            "Duplicate ticker pairs (including reversed-direction duplicates) found in output"
        )

    def test_canonical_order_is_deterministic(self, synthetic_data):
        """Two identical calls to score_candidates produce identical ticker ordering."""
        d = synthetic_data
        kwargs = dict(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        result1 = score_candidates(**kwargs)
        result2 = score_candidates(**kwargs)
        assert_frame_equal(result1, result2, check_exact=False, rtol=1e-5)


# ---------------------------------------------------------------------------
# Hard gate tests
# ---------------------------------------------------------------------------


class TestScoreCandidatesHardGates:

    def test_output_is_subset_of_candidate_pairs(self, synthetic_data):
        """Every output pair (in either direction) is a valid within-cluster candidate pair."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        candidate_pairs = build_candidate_pairs(d["clusters"])
        candidate_set = {
            frozenset([row["ticker_a"], row["ticker_b"]])
            for _, row in candidate_pairs.iterrows()
        }
        for _, row in result.iterrows():
            pair = frozenset([row["ticker_a"], row["ticker_b"]])
            assert pair in candidate_set, (
                f"Output pair ({row['ticker_a']}, {row['ticker_b']}) "
                f"is not a valid within-cluster candidate pair"
            )

    def test_composite_score_reflects_multiple_components(self, synthetic_data):
        """Pairs surviving gating in the same multi-pair cluster have differentiated scores."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if len(result) < 2:
            pytest.skip(
                "Fewer than 2 pairs survived gating — cannot test score differentiation"
            )
        for cluster_id, group in result.groupby("cluster_id"):
            if len(group) >= 2:
                scores = group["composite_score"].values
                assert not np.allclose(scores, scores[0]), (
                    f"All composite scores in cluster {cluster_id} are identical: {scores}"
                )
                break


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestScoreCandidatesIntegration:

    def test_full_pipeline_runs_without_error(self, synthetic_data):
        """score_candidates completes without raising any exception given valid synthetic data."""
        d = synthetic_data
        score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )

    def test_result_is_reproducible(self, synthetic_data):
        """Two identical calls return identical DataFrames within floating-point tolerance."""
        d = synthetic_data
        kwargs = dict(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        result1 = score_candidates(**kwargs)
        result2 = score_candidates(**kwargs)
        assert_frame_equal(result1, result2, check_exact=False, rtol=1e-5)

    def test_cluster_ids_in_output_match_input(self, synthetic_data):
        """All cluster_id values in the output are keys in the input clusters dict."""
        d = synthetic_data
        result = score_candidates(
            clusters=d["clusters"],
            returns=d["returns"],
            prices=d["prices"],
            as_of=d["as_of"],
        )
        if result.empty:
            pytest.skip("No pairs survived gating with synthetic data")
        assert set(result["cluster_id"]).issubset(set(d["clusters"].keys()))

    def test_larger_cluster_contributes_more_candidates_before_gating(self, synthetic_data):
        """The 3-ticker cluster generates 3 candidate pairs vs 1 for the 2-ticker cluster."""
        d = synthetic_data
        candidate_pairs = build_candidate_pairs(d["clusters"])
        cluster_0_count = int((candidate_pairs["cluster_id"] == 0).sum())
        cluster_1_count = int((candidate_pairs["cluster_id"] == 1).sum())
        assert cluster_0_count == 3, f"Expected 3 pairs from C(3,2), got {cluster_0_count}"
        assert cluster_1_count == 1, f"Expected 1 pair from C(2,2), got {cluster_1_count}"
        assert cluster_0_count > cluster_1_count
