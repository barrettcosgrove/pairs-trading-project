# Simulates market-on-close order execution for a pairs trade.
# Short leg is always submitted first. If the short leg fills below
# CONFIG.partial_fill_threshold, the long leg is cancelled and the
# order is queued for retry the following trading day.

import logging
from datetime import date

import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)


def execute(order: dict, prices: pd.DataFrame, as_of: date) -> dict:
    """
    Simulate market-on-close execution of a pairs order.

    Submits the short leg first. If the short leg fills below
    CONFIG.partial_fill_threshold (95%) of the intended quantity,
    the long leg is cancelled to avoid an unhedged long position,
    and the order is flagged for next-day retry.

    Args:
        order: Dict containing:
            ticker_a       (str)   — leg A ticker
            ticker_b       (str)   — leg B ticker
            shares_a       (float) — signed intended shares for leg A
                                     (positive = buy, negative = sell short)
            shares_b       (float) — signed intended shares for leg B
            action         (str)   — "OPEN" or "CLOSE"
        prices: DataFrame with columns [date, ticker, adj_close].
        as_of: Execution date. Uses market-on-close price for this date.

    Returns:
        Dict containing:
            ticker_a        (str)   — leg A ticker
            ticker_b        (str)   — leg B ticker
            filled_a        (float) — actual shares filled for leg A (0 if cancelled)
            filled_b        (float) — actual shares filled for leg B
            price_a         (float) — fill price for leg A (0 if cancelled)
            price_b         (float) — fill price for leg B
            action          (str)   — echoed from order
            partial_fill    (bool)  — True if short leg filled below threshold
            retry_next_day  (bool)  — True if long leg was cancelled and needs retry
            success         (bool)  — True if both legs filled at intended quantity
    """
    ticker_a   = order["ticker_a"]
    ticker_b   = order["ticker_b"]
    shares_a   = order["shares_a"]
    shares_b   = order["shares_b"]
    action     = order["action"]

    # ── Market-on-close price lookup ──────────────────────────────────────────
    day_prices = prices[prices["date"] == pd.Timestamp(as_of)]
    price_map  = day_prices.set_index("ticker")["adj_close"].to_dict()

    if ticker_a not in price_map or ticker_b not in price_map:
        logger.warning(
            "Missing MOC price for %s or %s on %s — order not filled",
            ticker_a, ticker_b, as_of,
        )
        return _empty_result(ticker_a, ticker_b, action)

    price_a = price_map[ticker_a]
    price_b = price_map[ticker_b]

    # ── Determine which leg is short ──────────────────────────────────────────
    # Short leg is submitted first per execution spec
    a_is_short = shares_a < 0
    b_is_short = shares_b < 0

    if a_is_short:
        short_ticker, short_shares, short_price = ticker_a, shares_a, price_a
        long_ticker,  long_shares,  long_price  = ticker_b, shares_b, price_b
        short_is_a = True
    else:
        short_ticker, short_shares, short_price = ticker_b, shares_b, price_b
        long_ticker,  long_shares,  long_price  = ticker_a, shares_a, price_a
        short_is_a = False

    # ── Simulate short leg fill ───────────────────────────────────────────────
    # In simulation all orders fill at MOC. Partial fills modelled as a
    # Bernoulli event — in a real broker integration this would come from
    # actual fill confirmations.
    short_fill_ratio = _simulate_fill_ratio(abs(short_shares), short_price)
    short_filled     = short_shares * short_fill_ratio

    logger.debug(
        "Short leg %s — intended %.2f shares, filled %.2f (ratio %.3f)",
        short_ticker, short_shares, short_filled, short_fill_ratio,
    )

    # ── Partial fill check ────────────────────────────────────────────────────
    if short_fill_ratio < CONFIG.partial_fill_threshold:
        logger.warning(
            "Partial fill on short leg %s — %.1f%% filled (threshold %.1f%%). "
            "Cancelling long leg %s and queuing retry.",
            short_ticker, short_fill_ratio * 100,
            CONFIG.partial_fill_threshold * 100, long_ticker,
        )
        if short_is_a:
            return {
                "ticker_a": ticker_a, "ticker_b": ticker_b,
                "filled_a": short_filled, "filled_b": 0.0,
                "price_a": short_price,  "price_b": 0.0,
                "action": action,
                "partial_fill": True, "retry_next_day": True, "success": False,
            }
        else:
            return {
                "ticker_a": ticker_a, "ticker_b": ticker_b,
                "filled_a": 0.0,          "filled_b": short_filled,
                "price_a": 0.0,           "price_b": short_price,
                "action": action,
                "partial_fill": True, "retry_next_day": True, "success": False,
            }

    # ── Short leg filled — submit long leg ────────────────────────────────────
    long_filled = long_shares  # long leg always fills at MOC in simulation

    logger.info(
        "Executed %s %s/%s — short %s %.2f @ $%.2f, long %s %.2f @ $%.2f",
        action, ticker_a, ticker_b,
        short_ticker, short_filled, short_price,
        long_ticker,  long_filled,  long_price,
    )

    if short_is_a:
        return {
            "ticker_a": ticker_a, "ticker_b": ticker_b,
            "filled_a": short_filled, "filled_b": long_filled,
            "price_a": short_price,   "price_b": long_price,
            "action": action,
            "partial_fill": False, "retry_next_day": False, "success": True,
        }
    else:
        return {
            "ticker_a": ticker_a, "ticker_b": ticker_b,
            "filled_a": long_filled,  "filled_b": short_filled,
            "price_a": long_price,    "price_b": short_price,
            "action": action,
            "partial_fill": False, "retry_next_day": False, "success": True,
        }


def _simulate_fill_ratio(shares: float, price: float) -> float:
    """
    Return the simulated fill ratio for an order.

    In this daily-resolution simulation all MOC orders fill at 100%
    unless the dollar value of the order exceeds a size threshold
    that would realistically cause market impact. For the current
    universe and position sizes this always returns 1.0, but the
    hook exists for future extension.

    Args:
        shares: Absolute share count of the order.
        price:  MOC price in USD.

    Returns:
        Fill ratio in [0, 1]. Returns 1.0 in all current cases.
    """
    return 1.0


def _empty_result(ticker_a: str, ticker_b: str, action: str) -> dict:
    """
    Return a zeroed fill result for orders that cannot be executed.

    Args:
        ticker_a: Leg A ticker.
        ticker_b: Leg B ticker.
        action: Order action string.

    Returns:
        Fill result dict with all quantities and prices at zero.
    """
    return {
        "ticker_a": ticker_a, "ticker_b": ticker_b,
        "filled_a": 0.0, "filled_b": 0.0,
        "price_a":  0.0, "price_b":  0.0,
        "action": action,
        "partial_fill": False, "retry_next_day": False, "success": False,
    }