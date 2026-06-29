"""
src/scoring/halflife.py — Spread Half-Life Scoring

Estimates mean-reversion speed for each candidate pair by fitting an AR(1)
regression to the OLS-adjusted spread. Half-life is computed in both spread
directions to determine canonical ordering (which ticker is stock A).
Returns a binary score: 1.0 if the best-direction half-life falls within
[CONFIG.halflife_min, CONFIG.halflife_max], 0.0 otherwise.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

REQUIRED_RETURN_COLUMNS = {"date", "ticker", "log_return"}
REQUIRED_CANDIDATE_PAIR_COLUMNS = {"ticker_a", "ticker_b", "cluster_id"}


def score(
    ticker_a: str,
    ticker_b: str,
    returns: pd.DataFrame,
    as_of: date,
) -> float:
    """
    Score a pair's spread half-life against the configured acceptable range.

    Returns 1.0 if the best-direction half-life falls within
    [CONFIG.halflife_min, CONFIG.halflife_max]. Returns 0.0 for diverging pairs,
    pairs with insufficient history, or half-life outside the acceptable range.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Float score of 0.0 or 1.0.
    """
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)
    score_val, _best_hl = _score_from_wide(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        returns_wide=returns_wide,
    )
    return float(score_val)


def canonical_order(
    ticker_a: str,
    ticker_b: str,
    returns: pd.DataFrame,
    as_of: date,
) -> str:
    """
    Determine which ticker should be stock A based on half-life direction.

    Computes the spread half-life in both directions. The direction with the
    smaller finite half-life determines the canonical ordering. Stock A is
    the ticker that appears first in the winning direction.

    Args:
        ticker_a: First ticker.
        ticker_b: Second ticker.
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical scoring date.

    Returns:
        'ab' if ticker_a should remain stock A.
        'ba' if ticker_b should become stock A.
        'ab' as default when both directions give equal or infinite half-lives.
    """
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)
    ticker_lo, ticker_hi = sorted([ticker_a, ticker_b])
    pair_returns = _extract_pair_returns(ticker_lo, ticker_hi, returns_wide)
    trailing = pair_returns.tail(CONFIG.signal_window)

    if len(trailing) < CONFIG.halflife_min + 2:
        return "ab"

    log_lo = trailing[ticker_lo].cumsum()
    log_hi = trailing[ticker_hi].cumsum()

    # OLS for forward: log_lo = alpha + beta * log_hi
    X_fwd = np.column_stack([np.ones(len(log_hi)), log_hi.values])
    coeffs_fwd, _, _, _ = np.linalg.lstsq(X_fwd, log_lo.values, rcond=None)
    beta_fwd = coeffs_fwd[1]

    # OLS for reverse: log_hi = alpha + beta * log_lo
    X_rev = np.column_stack([np.ones(len(log_lo)), log_lo.values])
    coeffs_rev, _, _, _ = np.linalg.lstsq(X_rev, log_hi.values, rcond=None)
    beta_rev = coeffs_rev[1]

    spread_forward = log_lo - beta_fwd * log_hi
    spread_reverse = log_hi - beta_rev * log_lo
    spread_forward = spread_forward - spread_forward.mean()
    spread_reverse = spread_reverse - spread_reverse.mean()

    hl_forward = _compute_halflife(spread_forward)
    hl_reverse = _compute_halflife(spread_reverse)

    forward_finite = np.isfinite(hl_forward)
    reverse_finite = np.isfinite(hl_reverse)

    if not forward_finite and not reverse_finite:
        return "ab"
    if forward_finite and not reverse_finite:
        # forward wins: spread is ticker_lo - ticker_hi
        # ticker_lo is stock A
        return "ab" if ticker_lo == ticker_a else "ba"
    if reverse_finite and not forward_finite:
        # reverse wins: spread is ticker_hi - ticker_lo
        # ticker_hi is stock A
        return "ab" if ticker_hi == ticker_a else "ba"
    # Both finite: take the smaller halflife
    if hl_forward <= hl_reverse:
        return "ab" if ticker_lo == ticker_a else "ba"
    else:
        return "ab" if ticker_hi == ticker_a else "ba"


def score_candidate_pairs(
    candidate_pairs: pd.DataFrame,
    returns: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Score half-life and determine canonical ordering for a batch of candidate pairs.

    Args:
        candidate_pairs: DataFrame with columns [ticker_a, ticker_b, cluster_id].
        returns: Long-form return DataFrame with columns [date, ticker, log_return].
        as_of: Historical scoring date. Only data on or before this date is
            used, preventing look-ahead bias.

    Returns:
        Copy of candidate_pairs with two added columns:
            halflife_score: 1.0 if the best half-life is in
                [CONFIG.halflife_min, CONFIG.halflife_max], 0.0 otherwise.
            halflife_canonical_order: 'ab' or 'ba' indicating which ticker
                should be stock A for downstream signal generation.
    """
    _validate_candidate_pairs(candidate_pairs)
    returns_wide = _build_returns_matrix(returns=returns, as_of=as_of)

    scored_pairs = candidate_pairs.copy()
    # Apply returns a Series of tuples
    hl_results = scored_pairs.apply(
        lambda row: _score_from_wide(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            returns_wide=returns_wide,
        ),
        axis=1,
    )
    
    scored_pairs["halflife_score"] = [r[0] for r in hl_results]
    scored_pairs["halflife_value"] = [r[1] for r in hl_results]
    scored_pairs["halflife_canonical_order"] = scored_pairs.apply(
        lambda row: canonical_order(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
            returns=returns,
            as_of=as_of,
        ),
        axis=1,
    )

    passing = (scored_pairs["halflife_score"] == 1.0).sum()
    logger.info(
        "Scored half-life for %d candidate pairs as of %s (%d pass gate)",
        len(scored_pairs),
        as_of,
        passing,
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


def _compute_halflife(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via AR(1) regression on a demeaned spread.

    Uses numpy.linalg.lstsq rather than statsmodels — sufficient for a
    two-parameter regression and avoids a heavyweight dependency for a single
    OLS call.

    Args:
        spread: Demeaned spread series indexed by date.

    Returns:
        Half-life in trading days. Returns inf if the AR(1) slope is
        non-negative (diverging pair), if fewer than two regression
        observations remain after differencing, or if the regression fails.
    """
    try:
        delta = spread.diff().dropna()
        lag = spread.shift(1).dropna()

        common_idx = delta.index.intersection(lag.index)
        delta = delta.loc[common_idx]
        lag = lag.loc[common_idx]

        if len(delta) < 2:
            return float("inf")

        X = np.column_stack([np.ones(len(lag)), lag.values])
        coeffs, _, _, _ = np.linalg.lstsq(X, delta.values, rcond=None)
        beta = coeffs[1]

        if beta >= 0:
            return float("inf")

        return float(np.log(2) / abs(beta))

    except Exception:
        return float("inf")


def _score_from_wide(
    ticker_a: str,
    ticker_b: str,
    returns_wide: pd.DataFrame,
) -> tuple[float, float]:
    """
    Score the spread half-life for one pair using a pre-built return matrix.

    Sorts tickers alphabetically for symmetry, computes the cumulative log-return
    spread in both directions, takes the minimum finite half-life, and returns
    1.0 if it falls within [CONFIG.halflife_min, CONFIG.halflife_max].

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.
        returns_wide: Wide date-by-ticker return matrix containing only data
            available on or before the scoring date.

    Returns:
        Tuple of (score, best_hl):
        score is 1.0 if the minimum finite half-life falls within [CONFIG.halflife_min,
        CONFIG.halflife_max]. 0.0 otherwise.
        best_hl is the raw expected half-life in days.
    """
    # Sort alphabetically so score(A, B) == score(B, A) regardless of call order.
    ticker_lo, ticker_hi = sorted([ticker_a, ticker_b])

    pair_returns = _extract_pair_returns(
        ticker_a=ticker_lo,
        ticker_b=ticker_hi,
        returns_wide=returns_wide,
    )

    trailing = pair_returns.tail(CONFIG.signal_window)

    if len(trailing) < CONFIG.halflife_min + 2:
        logger.debug(
            "Returning 0.0 for %s/%s: only %d aligned observations (minimum %d)",
            ticker_lo,
            ticker_hi,
            len(trailing),
            CONFIG.halflife_min + 2,
        )
        return 0.0, float("inf")

    log_lo = trailing[ticker_lo].cumsum()
    log_hi = trailing[ticker_hi].cumsum()

    # OLS for forward: log_lo = alpha + beta * log_hi
    X_fwd = np.column_stack([np.ones(len(log_hi)), log_hi.values])
    coeffs_fwd, _, _, _ = np.linalg.lstsq(X_fwd, log_lo.values, rcond=None)
    beta_fwd = coeffs_fwd[1]

    # OLS for reverse: log_hi = alpha + beta * log_lo
    X_rev = np.column_stack([np.ones(len(log_lo)), log_lo.values])
    coeffs_rev, _, _, _ = np.linalg.lstsq(X_rev, log_hi.values, rcond=None)
    beta_rev = coeffs_rev[1]

    spread_forward = log_lo - beta_fwd * log_hi
    spread_reverse = log_hi - beta_rev * log_lo
    spread_forward = spread_forward - spread_forward.mean()
    spread_reverse = spread_reverse - spread_reverse.mean()

    hl_forward = _compute_halflife(spread_forward)
    hl_reverse = _compute_halflife(spread_reverse)

    best_hl = min(hl_forward, hl_reverse)

    if not np.isfinite(best_hl):
        logger.debug(
            "Returning 0.0 for %s/%s: non-finite half-life (diverging or regression failure)",
            ticker_lo,
            ticker_hi,
        )
        return 0.0, float("inf")

    if CONFIG.halflife_min <= best_hl <= CONFIG.halflife_max:
        return 1.0, best_hl

    logger.debug(
        "Returning 0.0 for %s/%s: half-life %.1f days outside [%d, %d]",
        ticker_lo,
        ticker_hi,
        best_hl,
        CONFIG.halflife_min,
        CONFIG.halflife_max,
    )
    return 0.0, best_hl
