"""
src/scoring/volatility.py - Volatility Compatibility Scoring

Scores how compatible a pair's volatility profiles are over two rolling
windows. A lower combined volatility ratio (larger/smaller annualised vol)
yields a higher score. The module is compute-only and is intended to be
called by composite.py for one pair at a time or across a batch of candidate
pairs.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

REQUIRED_RETURN_COLUMNS = {"date", "ticker", "log_return"}
REQUIRED_CANDIDATE_PAIR_COLUMNS = {"ticker_a", "ticker_b", "cluster_id"}

_ANNUALISATION_FACTOR = np.sqrt(252)


def score(ticker_a: str, ticker_b: str, returns: pd.DataFrame, as_of: date) -> float:
    """
    Score the volatility compatibility of a pair over two rolling windows.

    Computes annualised volatility for each ticker over a short window
    (CONFIG.volatility_short_window) and a long window
    (CONFIG.volatility_long_window), forms the ratio max/min for each window,
    combines them with CONFIG.volatility_short_weight and
    CONFIG.volatility_long_weight, then returns 1.0 / combined_ratio.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Float in [0, 1]. Returns 0.0 when fewer than
        CONFIG.volatility_long_window aligned observations exist, when the
        short-window ratio exceeds CONFIG.max_volatility_ratio, or when any
        vol computation fails.
    """
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)
    return _score_from_wide(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        returns_wide=returns_wide,
    )


def score_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    returns: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Score volatility compatibility across a batch of candidate pairs.

    Args:
        candidate_pairs: DataFrame with columns [ticker_a, ticker_b, cluster_id].
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Copy of candidate_pairs with an added volatility_score column.
    """
    _validate_candidate_pairs(candidate_pairs)
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)

    scored_pairs = candidate_pairs.copy()
    scored_pairs["volatility_score"] = scored_pairs.apply(
        lambda row: _score_from_wide(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            returns_wide=returns_wide,
        ),
        axis=1,
    )

    logger.info(
        "Scored volatility compatibility for %d candidate pairs as of %s",
        len(scored_pairs),
        as_of,
    )

    return scored_pairs


def _validate_returns_columns(returns: pd.DataFrame) -> None:
    """
    Validate the returns DataFrame against the scoring input contract.

    Args:
        returns: Candidate long-form return DataFrame.

    Returns:
        None. Raises ValueError if required columns are missing.
    """
    missing_columns = REQUIRED_RETURN_COLUMNS.difference(returns.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Returns DataFrame is missing required columns: {missing_list}")


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


def _build_returns_matrix(returns: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """
    Build a date-by-ticker return matrix using only data available by as_of.

    Args:
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical cutoff date used to prevent look-ahead bias.

    Returns:
        Wide DataFrame indexed by date with ticker columns and log_return values.
    """
    _validate_returns_columns(returns)

    returns_filtered = returns.copy()
    returns_filtered["date"] = pd.to_datetime(returns_filtered["date"])
    returns_filtered = returns_filtered[returns_filtered["date"] <= pd.Timestamp(as_of)]
    returns_filtered = returns_filtered.sort_values(["date", "ticker"]).reset_index(drop=True)

    if returns_filtered.empty:
        return pd.DataFrame()

    return returns_filtered.pivot(index="date", columns="ticker", values="log_return")


def _extract_pair_returns(
    ticker_a: str,
    ticker_b: str,
    returns_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract aligned return observations for a single ticker pair.

    Args:
        ticker_a: First ticker in the pair.
        ticker_b: Second ticker in the pair.
        returns_wide: Wide date-by-ticker return matrix.

    Returns:
        Two-column DataFrame of aligned return observations for the pair. The
        result is empty if one or both tickers are unavailable.
    """
    if returns_wide.empty:
        return pd.DataFrame(columns=[ticker_a, ticker_b])

    if ticker_a not in returns_wide.columns or ticker_b not in returns_wide.columns:
        return pd.DataFrame(columns=[ticker_a, ticker_b])

    return returns_wide[[ticker_a, ticker_b]].dropna()


def _compute_volatility_ratio(series_a: pd.Series, series_b: pd.Series) -> float | None:
    """
    Compute the annualised volatility ratio for a pair over a given window.

    The ratio is always max(vol_a, vol_b) / min(vol_a, vol_b), so it is
    symmetric: swapping A and B produces the same result.

    Args:
        series_a: Log-return series for the first ticker.
        series_b: Log-return series for the second ticker.

    Returns:
        Float >= 1.0 representing the volatility ratio. Returns None if either
        series has zero or near-zero volatility, which would make the ratio
        undefined or degenerate.
    """
    vol_a = float(series_a.std() * _ANNUALISATION_FACTOR)
    vol_b = float(series_b.std() * _ANNUALISATION_FACTOR)

    if vol_a <= 0.0 or vol_b <= 0.0:
        return None

    return max(vol_a, vol_b) / min(vol_a, vol_b)


def _score_from_wide(
    ticker_a: str,
    ticker_b: str,
    returns_wide: pd.DataFrame,
) -> float:
    """
    Score volatility compatibility for one pair using a pre-built return matrix.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns_wide: Wide date-by-ticker return matrix containing only data
            available on or before the scoring date.

    Returns:
        Float in [0, 1]. Returns 0.0 when history is insufficient, the
        short-window ratio exceeds CONFIG.max_volatility_ratio, or any vol
        computation fails.
    """
    pair_returns = _extract_pair_returns(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        returns_wide=returns_wide,
    )

    if len(pair_returns) < CONFIG.volatility_long_window:
        logger.debug(
            "Returning 0.0 for %s/%s: only %d aligned observations (minimum %d)",
            ticker_a,
            ticker_b,
            len(pair_returns),
            CONFIG.volatility_long_window,
        )
        return 0.0

    short_window = pair_returns.tail(CONFIG.volatility_short_window)
    long_window = pair_returns.tail(CONFIG.volatility_long_window)

    # max/min ratio is symmetric: swapping ticker_a and ticker_b yields the
    # same value, so no explicit alphabetical sort is needed here.
    short_ratio = _compute_volatility_ratio(
        series_a=short_window[ticker_a],
        series_b=short_window[ticker_b],
    )
    long_ratio = _compute_volatility_ratio(
        series_a=long_window[ticker_a],
        series_b=long_window[ticker_b],
    )

    if short_ratio is None or long_ratio is None:
        logger.debug(
            "Returning 0.0 for %s/%s: volatility computation failed (zero vol)",
            ticker_a,
            ticker_b,
        )
        return 0.0

    if short_ratio > CONFIG.max_volatility_ratio:
        logger.debug(
            "Returning 0.0 for %s/%s: short-window ratio %.2f exceeds hard gate %.2f",
            ticker_a,
            ticker_b,
            short_ratio,
            CONFIG.max_volatility_ratio,
        )
        return 0.0

    combined_ratio = (
        CONFIG.volatility_short_weight * short_ratio
        + CONFIG.volatility_long_weight * long_ratio
    )

    return float(np.clip(1.0 / combined_ratio, 0.0, 1.0))
