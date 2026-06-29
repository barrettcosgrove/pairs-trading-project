# Takes composite score finalists and confirms each pair via Johansen cointegration.
# Returns the top CONFIG.max_pairs (2) confirmed pairs as a flat list, ranked by
# composite score. No tiering — single pool, equal capital allocation.

import logging
from datetime import date

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import CONFIG

logger = logging.getLogger(__name__)


def _johansen_p_value(ticker_a: str, ticker_b: str, prices: pd.DataFrame) -> float:
    """
    Run Johansen trace test on the log prices of a pair.

    Args:
        ticker_a: Ticker symbol for leg A.
        ticker_b: Ticker symbol for leg B.
        prices: DataFrame with columns [date, ticker, adj_close].

    Returns:
        Approximate p-value from the Johansen trace statistic at rank 0.
        Returns 1.0 if the test cannot be run (insufficient data).
    """
    pivot = prices.pivot(index="date", columns="ticker", values="adj_close")
    log_prices = np.log(pivot[[ticker_a, ticker_b]].dropna())

    if len(log_prices) < CONFIG.signal_window:
        logger.warning(
            "Insufficient data for Johansen test on %s/%s (%d rows)",
            ticker_a, ticker_b, len(log_prices),
        )
        return 1.0

    try:
        result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        crit_90 = result.cvt[0, 0]
        crit_95 = result.cvt[0, 1]
        crit_99 = result.cvt[0, 2]

        if trace_stat > crit_99:
            return 0.01
        elif trace_stat > crit_95:
            return 0.05
        elif trace_stat > crit_90:
            return 0.10
        else:
            return 0.20
    except Exception as exc:
        logger.warning("Johansen test failed for %s/%s: %s", ticker_a, ticker_b, exc)
        return 1.0


def select_pairs(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> list[tuple[str, str]]:
    """
    Select the top CONFIG.max_pairs confirmed pairs from scored candidates.

    Iterates candidates in composite score order (descending). Each pair is
    confirmed via Johansen cointegration — pairs that fail are skipped.
    Stops once CONFIG.max_pairs (2) confirmed pairs are found.

    Args:
        candidates: DataFrame with columns [ticker_a, ticker_b, cluster_id,
                    composite_score], sorted by composite_score descending.
        prices: DataFrame with columns [date, ticker, adj_close].
        as_of: Selection date. Only data on or before this date is used.

    Returns:
        List of up to CONFIG.max_pairs (ticker_a, ticker_b) tuples,
        ordered by composite score descending. May be shorter than
        max_pairs if fewer candidates pass Johansen.
    """
    prices_as_of = prices[prices["date"] <= pd.Timestamp(as_of)]
    selected: list[tuple[str, str]] = []

    for _, row in candidates.iterrows():
        if len(selected) >= CONFIG.max_pairs:
            break

        ticker_a = row["ticker_a"]
        ticker_b = row["ticker_b"]

        j_pval = _johansen_p_value(ticker_a, ticker_b, prices_as_of)

        if j_pval < CONFIG.johansen_threshold:
            logger.info(
                "Pair %s/%s confirmed (Johansen p=%.3f) — rank %d of %d",
                ticker_a, ticker_b, j_pval, len(selected) + 1, CONFIG.max_pairs,
            )
            selected.append((ticker_a, ticker_b))
        else:
            logger.info(
                "Pair %s/%s failed Johansen (p=%.3f) — skipping",
                ticker_a, ticker_b, j_pval,
            )

    logger.info(
        "Pair selection complete as of %s — %d/%d pairs confirmed",
        as_of, len(selected), CONFIG.max_pairs,
    )

    return selected