"""
src/clustering/kmeans.py - Silhouette-Scored K-Means Clustering

Runs K-means across the correlation-distance representation of the investable
universe, selects the cluster count with the strongest silhouette score, and
returns ticker assignments grouped by cluster id.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import CONFIG

logger = logging.getLogger(__name__)


def run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]:
    """
    Cluster tickers using K-means on their distance-matrix feature vectors.

    Args:
        distance_matrix: NxN DataFrame whose index and columns are ticker
            strings and whose values are pairwise distances in [0, 2].

    Returns:
        Dictionary mapping cluster ids to lists of ticker strings. Every input
        ticker appears in exactly one cluster, and no returned cluster is empty.
    """
    _validate_distance_matrix(distance_matrix)

    tickers = distance_matrix.index.tolist()
    feature_matrix = distance_matrix.to_numpy(dtype=float)
    k_min, k_max = _get_k_search_bounds(n_tickers=len(tickers))

    best_score = float("-inf")
    best_labels: np.ndarray | None = None
    best_k: int | None = None

    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = _fit_kmeans(feature_matrix=feature_matrix, k=k)
        score = _score_clustering(feature_matrix=feature_matrix, labels=labels)
        scores[k] = score
        logger.debug("k=%d silhouette=%.4f", k, score)

        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None or best_k is None:
        raise ValueError("Failed to produce any valid clustering result")

    clusters = _labels_to_cluster_dict(labels=best_labels, tickers=tickers)

    logger.info(
        "Selected k=%d with silhouette score %.6f across %d tickers",
        best_k,
        best_score,
        len(tickers),
    )

    return clusters


def _validate_distance_matrix(distance_matrix: pd.DataFrame) -> None:
    """
    Validate that the distance matrix satisfies the clustering contract.

    Args:
        distance_matrix: Candidate input to run_clustering().

    Returns:
        None. Raises ValueError if the matrix is not a valid NxN distance
        matrix suitable for K-means clustering.
    """
    if distance_matrix.empty:
        raise ValueError("distance_matrix must not be empty")

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square")

    if not distance_matrix.index.equals(distance_matrix.columns):
        raise ValueError("distance_matrix index and columns must match exactly")

    if distance_matrix.index.has_duplicates or distance_matrix.columns.has_duplicates:
        raise ValueError("distance_matrix tickers must be unique")

    values = distance_matrix.to_numpy(dtype=float)

    if np.isnan(values).any():
        raise ValueError("distance_matrix must not contain NaN values")

    if not np.allclose(values, values.T):
        raise ValueError("distance_matrix must be symmetric")

    if not np.allclose(np.diag(values), 0.0):
        raise ValueError("distance_matrix diagonal must be all zeros")

    if np.any(values < 0.0) or np.any(values > 2.0):
        raise ValueError("distance_matrix values must lie within [0, 2]")

    min_required_tickers = CONFIG.k_min + 1
    if distance_matrix.shape[0] < min_required_tickers:
        raise ValueError(
            f"Need at least {min_required_tickers} tickers to evaluate "
            f"k in [{CONFIG.k_min}, {CONFIG.k_max}] with silhouette scoring; "
            f"received {distance_matrix.shape[0]}"
        )


def _get_k_search_bounds(n_tickers: int) -> tuple[int, int]:
    """
    Determine the feasible K-means search range for the current universe size.

    Args:
        n_tickers: Number of tickers available for clustering.

    Returns:
        Tuple of (k_min, k_max) bounds inclusive, constrained by both CONFIG
        and the silhouette-score requirement that k must be less than the
        number of observations.
    """
    k_min = CONFIG.k_min
    k_max = min(CONFIG.k_max, n_tickers - 1)

    if k_min > k_max:
        raise ValueError(
            f"No valid k range for {n_tickers} tickers using config bounds "
            f"[{CONFIG.k_min}, {CONFIG.k_max}]"
        )

    return k_min, k_max


def _fit_kmeans(feature_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Fit K-means for a single cluster count and return the assigned labels.

    Args:
        feature_matrix: Dense numeric matrix where each row is a ticker and
            each column is a distance-based feature.
        k: Number of clusters to fit.

    Returns:
        One-dimensional integer array of cluster labels aligned to the input
        row order.
    """
    model = KMeans(
        n_clusters=k,
        n_init=CONFIG.kmeans_restarts,
        random_state=CONFIG.random_seed,
    )
    return model.fit_predict(feature_matrix)


def _score_clustering(feature_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the average silhouette score for a fitted clustering assignment.

    Args:
        feature_matrix: Dense numeric matrix where each row is a ticker and
            each column is a distance-based feature.
        labels: Cluster labels produced by K-means.

    Returns:
        Average silhouette score as a float. Higher is better.
    """
    return float(silhouette_score(feature_matrix, labels, metric="euclidean"))


def _labels_to_cluster_dict(
    labels: np.ndarray,
    tickers: list[str],
) -> dict[int, list[str]]:
    """
    Convert row-aligned cluster labels into the module's public return format.

    Args:
        labels: Cluster labels aligned to the ticker order.
        tickers: Ticker symbols aligned to the label order.

    Returns:
        Dictionary mapping cluster ids to sorted ticker lists, with no empty
        clusters and every ticker assigned exactly once.
    """
    clusters: dict[int, list[str]] = {}

    for ticker, label in zip(tickers, labels, strict=True):
        cluster_id = int(label)
        clusters.setdefault(cluster_id, []).append(ticker)

    for cluster_id in clusters:
        clusters[cluster_id].sort()

    assigned_tickers = sorted(ticker for members in clusters.values() for ticker in members)
    if assigned_tickers != sorted(tickers):
        raise ValueError("Cluster assignments do not cover each input ticker exactly once")

    if any(len(members) == 0 for members in clusters.values()):
        raise ValueError("Cluster assignments must not contain empty clusters")

    return dict(sorted(clusters.items()))

