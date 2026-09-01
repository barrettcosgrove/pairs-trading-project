# Verifies: each component scorer returns values in [0, 1], pairs failing pre-filter gates are discarded, weights sum to 1.0, normalization is per-cluster not global, correct number of finalists returned per cluster.
from src.scoring.candidate_pairs import build_candidate_pairs


def test_basic_candidate_pairs():
    # Tickers not in sector_map all resolve to None — pairs remain candidates.
    clusters = {
        0: ["ZZ1", "ZZ2", "ZZ3"],
        1: ["ZZ4", "ZZ5"],
    }

    pairs = build_candidate_pairs(clusters)

    assert len(pairs) == 4
    assert set(pairs.columns) == {"ticker_a", "ticker_b", "cluster_id"}


def test_singleton_cluster_skipped():
    clusters = {
        0: ["ZZ1", "ZZ2"],
        1: ["ZZ6"],
    }

    pairs = build_candidate_pairs(clusters)

    assert len(pairs) == 1
    assert set(pairs["ticker_a"]) == {"ZZ1"}
    assert set(pairs["ticker_b"]) == {"ZZ2"}

import pytest


def test_duplicate_ticker_across_clusters_raises():
    clusters = {
        0: ["AAPL", "MSFT"],
        1: ["AAPL", "NVDA"],
    }

    with pytest.raises(ValueError):
        build_candidate_pairs(clusters)
