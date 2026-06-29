# Fits OLS regression log(P_A) = α + β·log(P_B) + ε over CONFIG.signal_window
# trading days ending on as_of. Returns (β, α). Flags sign flips and triggers
# a rebalance flag when β drifts more than CONFIG.beta_rebalance_threshold from entry.

import logging
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from src.config import CONFIG

logger = logging.getLogger(__name__)


def compute(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
    beta_at_entry: float | None = None,
) -> tuple[float, float, bool, bool]:
    """
    Estimate the hedge ratio β and intercept α via OLS on log prices.

    Fits: log(P_A) = α + β · log(P_B) + ε
    over the CONFIG.signal_window trading days ending on as_of.

    Args:
        ticker_a: Ticker symbol for the dependent leg (A).
        ticker_b: Ticker symbol for the independent leg (B).
        prices: DataFrame with columns [date, ticker, adj_close].
        as_of: Estimation date. Only data on or before this date is used.
        beta_at_entry: β recorded when the position was opened. Pass None
                       if no position is currently open. Used to compute
                       the rebalance flag and sign-flip flag.

    Returns:
        Tuple of (β, α, sign_flipped, rebalance_needed) where:
            β              — hedge ratio (slope)
            α              — intercept
            sign_flipped   — True if β has changed sign relative to beta_at_entry
            rebalance_needed — True if |β - beta_at_entry| > CONFIG.beta_rebalance_threshold
        Returns (nan, nan, False, False) if insufficient data.
    """
    prices_to_date = prices[prices["date"] <= pd.Timestamp(as_of)]
    pivot = prices_to_date.pivot(index="date", columns="ticker", values="adj_close")

    if ticker_a not in pivot.columns or ticker_b not in pivot.columns:
        logger.warning("Missing price data for %s or %s as of %s", ticker_a, ticker_b, as_of)
        return float("nan"), float("nan"), False, False

    log_prices = np.log(pivot[[ticker_a, ticker_b]].dropna()).tail(CONFIG.signal_window)

    if len(log_prices) < CONFIG.signal_window:
        logger.warning(
            "Insufficient data for hedge ratio %s/%s — %d rows (need %d)",
            ticker_a, ticker_b, len(log_prices), CONFIG.signal_window,
        )
        return float("nan"), float("nan"), False, False

    log_a = log_prices[ticker_a].values
    log_b = log_prices[ticker_b].values

    result = OLS(log_a, add_constant(log_b)).fit()
    alpha = result.params[0]
    beta = result.params[1]

    # ── Sign flip and rebalance flags ─────────────────────────────────────────
    sign_flipped = False
    rebalance_needed = False

    if beta_at_entry is not None:
        sign_flipped = (beta * beta_at_entry) < 0
        rebalance_needed = abs(beta - beta_at_entry) > CONFIG.beta_rebalance_threshold

        if sign_flipped:
            logger.warning(
                "Beta sign flip detected for %s/%s — entry %.3f, current %.3f",
                ticker_a, ticker_b, beta_at_entry, beta,
            )
        elif rebalance_needed:
            logger.info(
                "Rebalance triggered for %s/%s — entry β=%.3f, current β=%.3f, drift=%.3f",
                ticker_a, ticker_b, beta_at_entry, beta, abs(beta - beta_at_entry),
            )

    logger.debug("Hedge ratio %s/%s as of %s — β=%.4f, α=%.4f", ticker_a, ticker_b, as_of, beta, alpha)

    return beta, alpha, sign_flipped, rebalance_needed