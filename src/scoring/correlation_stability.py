"""
src/scoring/correlation_stability.py - Correlation Stability Scoring

Scores how stable a pair's recent return correlation has been relative to a
longer historical baseline. The module is compute-only and is intended to be
called by composite.py for one pair at a time or across a batch of candidate
pairs.
"""

import logging
from datetime import date

import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

REQUIRED_RETURN_COLUMNS = {"date", "ticker", "log_return"}
REQUIRED_CANDIDATE_PAIR_COLUMNS = {"ticker_a", "ticker_b", "cluster_id"}


def score(ticker_a: str, ticker_b: str, returns: pd.DataFrame, as_of: date) -> float:
    """
    Score the stability of a pair's recent correlation versus its baseline.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns: Long-form return DataFrame with columns
            [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Float score in [0, 1]. Returns 0.0 when the pair lacks enough aligned
        history or when the recent correlation is below
        CONFIG.min_recent_correlation.
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
    Score correlation stability across a batch of candidate pairs.

    Args:
        candidate_pairs: DataFrame with columns [ticker_a, ticker_b, cluster_id].
        returns: Long-form return DataFrame with columns
            [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Copy of candidate_pairs with an added correlation_stability_score
        column.
    """
    _validate_candidate_pairs(candidate_pairs)
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)

    scored_pairs = candidate_pairs.copy()
    scored_pairs["correlation_stability_score"] = scored_pairs.apply(
        lambda row: _score_from_wide(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            returns_wide=returns_wide,
        ),
        axis=1,
    )

    logger.info(
        "Scored correlation stability for %d candidate pairs as of %s",
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
        returns: Long-form return DataFrame with columns
            [date, ticker, log_return].
        as_of: Historical cutoff date used to prevent look-ahead bias.

    Returns:
        Wide DataFrame indexed by date with ticker columns and log_return
        values.
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


def _compute_pair_correlation(pair_returns: pd.DataFrame, window: int) -> float:
    """
    Compute the Pearson correlation for the trailing window of aligned returns.

    Args:
        pair_returns: Two-column aligned return DataFrame for one pair.
        window: Number of trailing observations to use.

    Returns:
        Pairwise Pearson correlation as a float. Returns NaN when there is not
        enough aligned history.
    """
    if len(pair_returns) < window:
        return float("nan")

    trailing_returns = pair_returns.tail(window)
    return float(trailing_returns.iloc[:, 0].corr(trailing_returns.iloc[:, 1]))


def _score_from_wide(
    ticker_a: str,
    ticker_b: str,
    returns_wide: pd.DataFrame,
) -> float:
    """
    Score correlation stability for one pair using a pre-built return matrix.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns_wide: Wide date-by-ticker return matrix containing only data
            available on or before the scoring date.

    Returns:
        Float score in [0, 1]. Returns 0.0 when history is insufficient or the
        recent correlation fails the minimum threshold.
    """
    pair_returns = _extract_pair_returns(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        returns_wide=returns_wide,
    )

    recent_correlation = _compute_pair_correlation(
        pair_returns=pair_returns,
        window=CONFIG.signal_window,
    )
    historical_correlation = _compute_pair_correlation(
        pair_returns=pair_returns,
        window=CONFIG.correlation_stability_historical_window,
    )

    if pd.isna(recent_correlation) or pd.isna(historical_correlation):
        logger.debug(
            "Returning 0.0 for %s/%s due to insufficient aligned history",
            ticker_a,
            ticker_b,
        )
        return 0.0

    if recent_correlation < CONFIG.min_recent_correlation:
        return 0.0

    stability_score = 1.0 - abs(recent_correlation - historical_correlation)
    stability_score = max(0.0, min(1.0, stability_score))

    return float(stability_score)
