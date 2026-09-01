"""
src/scoring/composite.py — Composite Pair Scoring

Orchestrates the five-component scoring pipeline for candidate pairs within
each cluster. Calls all five scorer modules, applies a half-life hard gate
and a soft cointegration-score floor, applies canonical ticker ordering,
computes a raw absolute weighted composite score, filters by
CONFIG.min_composite_score, drops non-positive formation β, and returns
exactly CONFIG.finalists_per_cluster pairs per cluster.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.scoring import (
    cointegration,
    correlation_stability,
    fundamentals,
    halflife,
    volatility,
)
from src.scoring.candidate_pairs import build_candidate_pairs

logger = logging.getLogger(__name__)

_OUTPUT_COLUMNS = [
    "ticker_a", 
    "ticker_b", 
    "cluster_id", 
    "composite_score", 
    "beta_formation", 
    "mean_formation", 
    "std_formation",
    "halflife_value"
]


def score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Score and rank candidate pairs within each cluster using a raw composite score.

    Generates all within-cluster candidate pairs, runs all five scorer modules,
    applies the half-life hard gate and a soft cointegration-score floor,
    applies canonical ticker ordering, computes a raw absolute weighted
    composite score without normalization, filters by
    CONFIG.min_composite_score, drops pairs with non-positive formation β,
    and returns exactly CONFIG.finalists_per_cluster pairs per surviving
    cluster.

    Args:
        clusters: Dictionary mapping cluster id to a list of ticker strings,
            as produced by run_clustering() in src/clustering/kmeans.py.
        returns: Long-form return DataFrame with columns
            [date, ticker, log_return].
        prices: Long-form price DataFrame with columns
            [date, ticker, adj_close].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        DataFrame with columns [ticker_a, ticker_b, cluster_id,
        composite_score]. ticker_a reflects canonical stock A after half-life
        direction ordering is applied. Sorted by cluster_id ascending, then
        composite_score descending within each cluster. Returns an empty
        DataFrame with the correct columns when no pairs survive scoring.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)

    if not clusters:
        logger.warning(
            "score_candidates called with empty clusters dict — returning empty result"
        )
        return empty

    candidate_pairs = build_candidate_pairs(clusters)

    if candidate_pairs.empty:
        logger.warning(
            "No candidate pairs generated from %d cluster(s) — returning empty result",
            len(clusters),
        )
        return empty

    logger.info(
        "Scoring %d candidate pair(s) from %d cluster(s) as of %s",
        len(candidate_pairs),
        len(clusters),
        as_of,
    )

    scored = correlation_stability.score_candidate_pairs(
        candidate_pairs=candidate_pairs,
        returns=returns,
        as_of=as_of,
    )
    scored = halflife.score_candidate_pairs(
        candidate_pairs=scored,
        returns=returns,
        as_of=as_of,
    )
    scored = cointegration.score_candidate_pairs(
        candidate_pairs=scored,
        prices=prices,
        as_of=as_of,
    )
    scored = volatility.score_candidate_pairs(
        candidate_pairs=scored,
        returns=returns,
        as_of=as_of,
    )
    scored = fundamentals.score_candidate_pairs(
        candidate_pairs=scored,
    )

    pre_gate_by_cluster = scored.groupby("cluster_id").size().to_dict()

    scored = _apply_hard_gates(scored)

    if scored.empty:
        logger.warning(
            "All %d candidate pair(s) failed hard gates as of %s — returning empty result",
            len(candidate_pairs),
            as_of,
        )
        return empty

    surviving_clusters = set(scored["cluster_id"].unique())
    for cluster_id, n_pre in sorted(pre_gate_by_cluster.items()):
        if cluster_id not in surviving_clusters:
            logger.warning(
                "Cluster %d: all %d pair(s) failed hard gates as of %s",
                cluster_id,
                n_pre,
                as_of,
            )

    scored = _apply_canonical_ordering(scored)

    scored["composite_score"] = (
        CONFIG.weight_correlation_stability * scored["correlation_stability_score"]
        + CONFIG.weight_cointegration * scored["cointegration_score"]
        + CONFIG.weight_halflife * scored["halflife_score"]
        + CONFIG.weight_volatility * scored["volatility_score"]
        + CONFIG.weight_fundamentals * scored["fundamentals_score"]
    )

    scored = _apply_minimum_threshold(scored)

    if scored.empty:
        logger.warning(
            "No pairs survived minimum composite threshold of %.2f as of %s "
            "— returning empty result",
            CONFIG.min_composite_score,
            as_of,
        )
        return empty

    # Formation stats on all threshold-passers so a negative-β top pick
    # does not empty the cluster when a lower-ranked pair is valid.
    formation_stats = scored.apply(
        lambda row: _compute_formation_stats(row, prices, as_of), axis=1
    )
    scored = pd.concat([scored, formation_stats], axis=1)
    scored = scored.dropna(subset=["beta_formation", "mean_formation", "std_formation"])

    n_neg_beta = int((scored["beta_formation"] <= CONFIG.min_formation_beta).sum())
    scored = scored[scored["beta_formation"] > CONFIG.min_formation_beta].copy()
    if n_neg_beta:
        logger.info(
            "Dropped %d pair(s) with formation β <= %.3f as of %s",
            n_neg_beta,
            CONFIG.min_formation_beta,
            as_of,
        )

    if scored.empty:
        logger.warning(
            "No pairs left after formation-β filter as of %s — returning empty result",
            as_of,
        )
        return empty

    finalists = (
        scored.sort_values("composite_score", ascending=False)
        .groupby("cluster_id", sort=False)
        .head(CONFIG.finalists_per_cluster)
    )

    result = (
        finalists[_OUTPUT_COLUMNS]
        .sort_values(["cluster_id", "composite_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    logger.info(
        "Composite scoring complete: %d finalist pair(s) across %d cluster(s) as of %s",
        len(result),
        result["cluster_id"].nunique(),
        as_of,
    )

    return result


def _apply_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop pairs that fail the half-life gate or the soft cointegration floor.

    A pair is removed if halflife_score == 0.0 (half-life outside
    [CONFIG.halflife_min, CONFIG.halflife_max] or insufficient history) or
    if cointegration_score is below CONFIG.min_cointegration_score.

    Args:
        df: Scored candidate-pair DataFrame containing at least
            halflife_score and cointegration_score columns.

    Returns:
        Filtered copy of df with gate failures removed.
    """
    halflife_failures = int((df["halflife_score"] == 0.0).sum())
    coint_failures = int(
        (df["cointegration_score"] < CONFIG.min_cointegration_score).sum()
    )

    logger.info(
        "Hard gates: %d pair(s) fail halflife_score == 0.0; "
        "%d pair(s) fail cointegration_score < %.2f",
        halflife_failures,
        coint_failures,
        CONFIG.min_cointegration_score,
    )

    passing = df[
        (df["halflife_score"] != 0.0)
        & (df["cointegration_score"] >= CONFIG.min_cointegration_score)
    ].copy()
    return passing


def _apply_canonical_ordering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Swap ticker_a and ticker_b where the half-life direction is reversed.

    For rows where halflife_canonical_order == 'ba', the values of ticker_a
    and ticker_b are swapped so that ticker_a always represents the canonical
    stock A in the mean-reversion spread for downstream signal generation.
    The halflife_canonical_order column is dropped after the swap is applied.

    Args:
        df: Scored candidate-pair DataFrame containing ticker_a, ticker_b,
            and halflife_canonical_order columns.

    Returns:
        Copy of df with ticker_a/ticker_b swapped where appropriate and the
        halflife_canonical_order column removed.
    """
    result = df.copy()
    swap_mask = result["halflife_canonical_order"] == "ba"
    n_swaps = int(swap_mask.sum())

    if n_swaps:
        result.loc[swap_mask, ["ticker_a", "ticker_b"]] = (
            result.loc[swap_mask, ["ticker_b", "ticker_a"]].values
        )
        logger.info(
            "Canonical ordering: swapped ticker_a/ticker_b for %d pair(s)", n_swaps
        )

    return result.drop(columns=["halflife_canonical_order"])


def _apply_minimum_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop pairs whose raw composite score falls below CONFIG.min_composite_score.

    Args:
        df: Scored candidate-pair DataFrame containing a composite_score column.

    Returns:
        Filtered copy of df with all below-threshold pairs removed.
    """
    passing = df[df["composite_score"] >= CONFIG.min_composite_score].copy()
    n_dropped = len(df) - len(passing)

    if n_dropped:
        logger.info(
            "Minimum threshold %.2f dropped %d pair(s); %d pair(s) remain",
            CONFIG.min_composite_score,
            n_dropped,
            len(passing),
        )

    return passing


def _compute_formation_stats(row: pd.Series, prices: pd.DataFrame, as_of: date) -> pd.Series:
    """
    Compute the locked cointegrating vector (beta, mean, std) over the formation window.
    """
    ticker_a = row["ticker_a"]
    ticker_b = row["ticker_b"]
    
    prices_filtered = prices[prices["date"] <= pd.Timestamp(as_of)]
    pivot = prices_filtered.pivot(index="date", columns="ticker", values="adj_close")
    
    if ticker_a not in pivot.columns or ticker_b not in pivot.columns:
        return pd.Series({"beta_formation": np.nan, "mean_formation": np.nan, "std_formation": np.nan})
        
    log_prices = np.log(pivot[[ticker_a, ticker_b]].dropna()).tail(CONFIG.formation_window)
    
    if len(log_prices) < 30:
        return pd.Series({"beta_formation": np.nan, "mean_formation": np.nan, "std_formation": np.nan})
        
    log_a = log_prices[ticker_a].values
    log_b = log_prices[ticker_b].values
    
    X = np.column_stack([np.ones(len(log_b)), log_b])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_a, rcond=None)
    beta = coeffs[1]
    
    spread = log_a - beta * log_b
    
    return pd.Series({
        "beta_formation": float(beta), 
        "mean_formation": float(np.mean(spread)), 
        "std_formation": float(np.std(spread))
    })
