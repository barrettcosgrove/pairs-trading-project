# Pair-level earnings blackout filter using end-of-quarter approximation.
# Blocks new entries during the last CONFIG.earnings_blackout_days_before
# trading days of March, June, September, and December, plus
# CONFIG.earnings_blackout_days_after trading days after quarter end.
# Per-company earnings dates are not used — documented simplification.

import logging
from datetime import date

import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

# Quarter-end months — last trading day of these months is the blackout anchor
_QUARTER_END_MONTHS = {3, 6, 9, 12}


def _quarter_end_dates(as_of: date) -> list[pd.Timestamp]:
    """
    Return the last trading day of each quarter-end month in the
    trailing 12 months relative to as_of.

    Args:
        as_of: Reference date.

    Returns:
        List of Timestamps representing quarter-end trading days.
    """
    anchors = []
    for year in [as_of.year - 1, as_of.year]:
        for month in _QUARTER_END_MONTHS:
            # Last calendar day of the quarter-end month
            last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
            # Roll back to the last business day if it falls on a weekend
            last_trading = last_day - pd.offsets.BDay(0)
            anchors.append(last_trading)
    return anchors


def in_blackout(ticker: str, as_of: date) -> bool:
    """
    Return whether a ticker is in an earnings blackout window on a given date.

    Uses end-of-quarter approximation rather than per-company earnings dates.
    Blackout covers the last CONFIG.earnings_blackout_days_before (5) trading
    days of each quarter-end month and CONFIG.earnings_blackout_days_after (1)
    trading day after the quarter end. Both tickers in a pair must be checked
    independently — the pair is blocked if either returns True.

    Args:
        ticker: Ticker symbol. Not used in logic (approximation applies
                uniformly) but retained in the signature for engine compatibility
                and future per-company extension.
        as_of: Date to evaluate. Only past quarter ends are considered.

    Returns:
        True if as_of falls within any earnings blackout window, False otherwise.
    """
    as_of_ts = pd.Timestamp(as_of)

    for quarter_end in _quarter_end_dates(as_of):
        # Blackout start: earnings_blackout_days_before trading days before quarter end
        blackout_start = quarter_end - pd.offsets.BDay(CONFIG.earnings_blackout_days_before)
        # Blackout end: earnings_blackout_days_after trading days after quarter end
        blackout_end = quarter_end + pd.offsets.BDay(CONFIG.earnings_blackout_days_after)

        if blackout_start <= as_of_ts <= blackout_end:
            logger.info(
                "Earnings blackout active for %s on %s "
                "(quarter end %s, window %s → %s)",
                ticker, as_of, quarter_end.date(),
                blackout_start.date(), blackout_end.date(),
            )
            return True

    return False