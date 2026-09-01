"""
src/scoring/cointegration.py — Johansen Cointegration Scoring

Scores each candidate pair by running the Johansen trace test on the pair's
log prices. Applies Benjamini-Hochberg FDR correction across all pairs in the
cluster. Returns a normalized score in [0, 1], where higher is better
(lower p-value = stronger evidence of cointegration).
"""

import logging
import math
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import CONFIG

logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "adj_close"}
REQUIRED_CANDIDATE_PAIR_COLUMNS = {"ticker_a", "ticker_b", "cluster_id"}

# Minimum aligned observations required for a reliable Johansen test
_MIN_JOHANSEN_OBS = 50


def score(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
) -> float:
    """
    Score a pair's cointegration strength using a bidirectional Johansen trace test.

    Runs the Johansen test in both column orderings ([A, B] and [B, A]) and
    averages the resulting p-values to remove ordering sensitivity. The averaged
    p-value is mapped to a score in [0, 1]: lower p-value → higher score.
    The averaged p-value is mapped continuously onto [0, 1]; it is not
    hard-zeroed at CONFIG.johansen_threshold.

    Note: This function scores a single pair without BH FDR correction. For
    batch scoring with per-cluster FDR correction, use score_candidate_pairs().

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        prices: Long-form price DataFrame with columns [date, ticker, adj_close].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Float score in [0, 1]. Returns 0.0 when the pair lacks sufficient
        aligned price history or the test cannot be computed.
    """
    _validate_price_columns(prices)

    pair_log_prices = _extract_log_prices(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        prices=prices,
        as_of=as_of,
    )

    if pair_log_prices is None:
        return 0.0

    p_forward = _run_johansen(pair_log_prices[[ticker_a, ticker_b]])
    p_reverse = _run_johansen(pair_log_prices[[ticker_b, ticker_a]])

    if math.isnan(p_forward) or math.isnan(p_reverse):
        logger.debug(
            "Returning 0.0 for %s/%s: Johansen test could not be computed",
            ticker_a,
            ticker_b,
        )
        return 0.0

    p_value = (p_forward + p_reverse) / 2.0

    return float(max(0.0, min(1.0, 1.0 - p_value)))


def score_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Score cointegration strength across a batch of candidate pairs with BH correction.

    Computes bidirectional Johansen p-values for all pairs, applies
    Benjamini-Hochberg FDR correction per cluster, then maps adjusted p-values
    to scores in [0, 1].

    Args:
        candidate_pairs: DataFrame with columns [ticker_a, ticker_b, cluster_id].
        prices: Long-form price DataFrame with columns [date, ticker, adj_close].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Copy of candidate_pairs with an added cointegration_score column.
        Pairs that fail the BH-corrected threshold receive a score of 0.0.
    """
    _validate_candidate_pairs(candidate_pairs)
    _validate_price_columns(prices)

    scored_pairs = candidate_pairs.copy()
    scored_pairs["_raw_pvalue"] = scored_pairs.apply(
        lambda row: _compute_raw_pvalue(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            prices=prices,
            as_of=as_of,
        ),
        axis=1,
    )

    adj_pvalue_series = scored_pairs.groupby("cluster_id")["_raw_pvalue"].transform(
        lambda x: pd.Series(_apply_bh_correction(x.values), index=x.index)
    )
    scored_pairs["cointegration_score"] = np.clip(
        1.0 - adj_pvalue_series, 0.0, 1.0
    )

    scored_pairs = scored_pairs.drop(columns=["_raw_pvalue"])

    logger.info(
        "Scored cointegration for %d candidate pairs as of %s",
        len(scored_pairs),
        as_of,
    )

    return scored_pairs


def _run_johansen(price_matrix: pd.DataFrame) -> float:
    """
    Run the Johansen trace test on a two-column log price matrix and return a p-value.

    Tests the null of no cointegration (rank = 0) against the alternative of at
    least one cointegrating vector. The p-value is approximated by piecewise
    linear interpolation from the statsmodels critical value table.

    Args:
        price_matrix: Two-column DataFrame of log prices, sorted ascending by date.
            Column order determines which series is treated as the first variable.

    Returns:
        Approximate p-value in [0.001, 1.0]. Returns float('nan') if the test
        raises a numerical exception.
    """
    try:
        result = coint_johansen(price_matrix.values, det_order=0, k_ar_diff=1)
    except Exception:
        logger.debug("Johansen test raised an exception", exc_info=True)
        return float("nan")

    trace_stat = float(result.lr1[0])
    cv10, cv5, cv1 = result.cvt[0]

    return _interpolate_johansen_pvalue(
        stat=trace_stat,
        cv10=float(cv10),
        cv5=float(cv5),
        cv1=float(cv1),
    )


def _interpolate_johansen_pvalue(
    stat: float,
    cv10: float,
    cv5: float,
    cv1: float,
) -> float:
    """
    Approximate a p-value from a Johansen trace statistic via critical value interpolation.

    Uses piecewise linear interpolation between the 10%, 5%, and 1% critical
    values. Linearly extrapolates below 0.01 for statistics that exceed the 1%
    critical value, flooring at 0.001. Statistics below the 10% critical value
    map continuously from p=1.0 (stat=0) to p=0.10 (stat=cv10).

    Args:
        stat: Johansen trace statistic for the null of no cointegration.
        cv10: Critical value at the 10% significance level.
        cv5: Critical value at the 5% significance level.
        cv1: Critical value at the 1% significance level.

    Returns:
        Approximate p-value in [0.001, 1.0].
    """
    if stat < cv10:
        if cv10 <= 0:
            return 1.0
        frac = max(0.0, min(1.0, stat / cv10))
        return 1.0 - frac * 0.90
    if stat < cv5:
        frac = (stat - cv10) / (cv5 - cv10)
        return 0.10 - frac * 0.05
    if stat < cv1:
        frac = (stat - cv5) / (cv1 - cv5)
        return 0.05 - frac * 0.04
    # Linearly extrapolate below 0.01 for strongly cointegrated pairs
    over = (stat - cv1) / cv1
    return max(0.001, 0.01 - over * 0.009)


def _compute_raw_pvalue(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
) -> float:
    """
    Compute the bidirectional average Johansen p-value for a single pair.

    Args:
        ticker_a: First ticker in the pair.
        ticker_b: Second ticker in the pair.
        prices: Long-form price DataFrame.
        as_of: Historical scoring cutoff date.

    Returns:
        Averaged p-value in [0.001, 1.0]. Returns 1.0 on any failure so that
        the BH correction treats the pair as non-cointegrated.
    """
    pair_log_prices = _extract_log_prices(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        prices=prices,
        as_of=as_of,
    )

    if pair_log_prices is None:
        return 1.0

    p_forward = _run_johansen(pair_log_prices[[ticker_a, ticker_b]])
    p_reverse = _run_johansen(pair_log_prices[[ticker_b, ticker_a]])

    if math.isnan(p_forward) or math.isnan(p_reverse):
        return 1.0

    return (p_forward + p_reverse) / 2.0


def _apply_bh_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Args:
        p_values: Array of raw p-values, one per candidate pair.

    Returns:
        Array of BH-adjusted p-values in the same order as the input.
    """
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)

    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]

    # BH-adjusted p_i = p_(i) * n / i, with monotonicity enforced from right
    adjusted = sorted_pvals * n / np.arange(1, n + 1)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    adjusted = np.minimum(adjusted, 1.0)

    result = np.empty(n, dtype=float)
    result[sorted_indices] = adjusted
    return result


def _extract_log_prices(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame | None:
    """
    Extract aligned log price observations for a ticker pair up to as_of.

    Args:
        ticker_a: First ticker in the pair.
        ticker_b: Second ticker in the pair.
        prices: Long-form price DataFrame with columns [date, ticker, adj_close].
        as_of: Historical cutoff date used to prevent look-ahead bias.

    Returns:
        Two-column DataFrame of log prices with columns [ticker_a, ticker_b],
        sorted ascending by date. Returns None if one or both tickers are
        absent or fewer than _MIN_JOHANSEN_OBS aligned observations exist.
    """
    prices_filtered = prices.copy()
    prices_filtered["date"] = pd.to_datetime(prices_filtered["date"])
    prices_filtered = prices_filtered[prices_filtered["date"] <= pd.Timestamp(as_of)]

    if prices_filtered.empty:
        return None

    prices_wide = prices_filtered.pivot(
        index="date", columns="ticker", values="adj_close"
    ).sort_index()

    if ticker_a not in prices_wide.columns or ticker_b not in prices_wide.columns:
        logger.debug(
            "Ticker(s) %s or %s not found in price history up to %s",
            ticker_a,
            ticker_b,
            as_of,
        )
        return None

    pair_prices = prices_wide[[ticker_a, ticker_b]].dropna().tail(CONFIG.johansen_window)

    if len(pair_prices) < _MIN_JOHANSEN_OBS:
        logger.debug(
            "Insufficient aligned observations for %s/%s: %d (need %d)",
            ticker_a,
            ticker_b,
            len(pair_prices),
            _MIN_JOHANSEN_OBS,
        )
        return None

    return np.log(pair_prices)


def _validate_price_columns(prices: pd.DataFrame) -> None:
    """
    Validate the prices DataFrame against the scoring input contract.

    Args:
        prices: Candidate long-form price DataFrame.

    Returns:
        None. Raises ValueError if required columns are missing.
    """
    missing_columns = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prices DataFrame is missing required columns: {missing_list}")


def _validate_candidate_pairs(candidate_pairs: pd.DataFrame) -> None:
    """
    Validate the candidate-pair batch input required for scoring.

    Args:
        candidate_pairs: Candidate pair DataFrame to validate.

    Returns:
        None. Raises ValueError if required columns are missing.
    """
    missing_columns = REQUIRED_CANDIDATE_PAIR_COLUMNS.difference(candidate_pairs.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Candidate pair DataFrame is missing required columns: {missing_list}"
        )
