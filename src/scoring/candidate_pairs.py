"""
src/scoring/candidate_pairs.py - Within-Cluster Candidate Pair Builder

Generates all unique within-cluster ticker pairs from the clustering output.
The resulting table is intended to feed composite.py, which will score and
rank pairs within each cluster.
"""

import logging
from itertools import combinations

import pandas as pd

from src.scoring.fundamentals import _get_sector_label

logger = logging.getLogger(__name__)


def build_candidate_pairs(clusters: dict[int, list[str]]) -> pd.DataFrame:
    """
    Build the set of unique within-cluster candidate pairs for scoring.

    Args:
        clusters: Dictionary mapping cluster ids to ticker lists, typically
            produced by run_clustering() in src/clustering/kmeans.py.

    Returns:
        DataFrame with columns [ticker_a, ticker_b, cluster_id]. Each row is a
        unique unordered ticker pair from a single cluster. Clusters with fewer
        than two tickers are skipped because they cannot form a pair.
    """
    _validate_clusters(clusters)

    rows: list[dict[str, object]] = []
    skipped_clusters = 0

    for cluster_id, tickers in sorted(clusters.items()):
        cluster_rows = _build_cluster_pairs(cluster_id=cluster_id, tickers=tickers)
        if not cluster_rows:
            skipped_clusters += 1
        rows.extend(cluster_rows)

    candidate_pairs = pd.DataFrame(
        rows,
        columns=["ticker_a", "ticker_b", "cluster_id"],
    )

    logger.info(
        "Built %d candidate pairs from %d clusters (%d singleton/empty cluster(s) skipped)",
        len(candidate_pairs),
        len(clusters),
        skipped_clusters,
    )

    return candidate_pairs


def _validate_clusters(clusters: dict[int, list[str]]) -> None:
    """
    Validate that the clustering output is suitable for pair generation.

    Args:
        clusters: Candidate cluster dictionary to validate.

    Returns:
        None. Raises ValueError if the cluster structure is invalid.
    """
    if not isinstance(clusters, dict):
        raise ValueError("clusters must be a dictionary of cluster_id -> ticker list")

    all_tickers: list[str] = []

    for cluster_id, tickers in clusters.items():
        if not isinstance(cluster_id, int):
            raise ValueError("cluster ids must be integers")

        if not isinstance(tickers, list):
            raise ValueError(f"Cluster {cluster_id} must map to a list of tickers")

        if any(not isinstance(ticker, str) or not ticker for ticker in tickers):
            raise ValueError(f"Cluster {cluster_id} contains an invalid ticker value")

        if len(set(tickers)) != len(tickers):
            raise ValueError(f"Cluster {cluster_id} contains duplicate tickers")

        all_tickers.extend(tickers)

    if len(set(all_tickers)) != len(all_tickers):
        raise ValueError("Each ticker must appear in exactly one cluster")


def _build_cluster_pairs(
    cluster_id: int,
    tickers: list[str],
) -> list[dict[str, object]]:
    """
    Build all unique unordered ticker pairs for a single cluster.

    Args:
        cluster_id: Cluster identifier associated with the ticker list.
        tickers: Tickers assigned to one cluster.

    Returns:
        List of dictionaries with keys [ticker_a, ticker_b, cluster_id]. Returns
        an empty list when the cluster has fewer than two tickers.
    """
    if len(tickers) < 2:
        logger.debug(
            "Skipping cluster %d because it contains fewer than two tickers",
            cluster_id,
        )
        return []

    sorted_tickers = sorted(tickers)

    valid_pairs = []
    for ticker_a, ticker_b in combinations(sorted_tickers, 2):
        if _get_sector_label(ticker_a) == _get_sector_label(ticker_b):
            valid_pairs.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "cluster_id": cluster_id,
            })
            
    return valid_pairs
