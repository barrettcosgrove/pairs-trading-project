# Applies the full transaction cost model to every trade.
# Commission, slippage, and bid-ask are applied to both legs on open and close.
# Short borrow accrues daily on the short leg while a position is held.
# Trades where expected profit < CONFIG.min_profit_to_cost_ratio × round-trip
# cost are skipped and logged.

import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)


def apply_costs(trade: dict) -> float:
    """
    Compute and return the total transaction cost for a trade.

    Applies commission, slippage, and bid-ask spread to both legs.
    Short borrow cost is computed separately via accrue_borrow_cost().
    Logs and returns -1.0 if expected profit fails the minimum
    profit-to-cost ratio gate — the engine must check for this sentinel
    and skip the trade.

    Args:
        trade: Dict containing:
            ticker_a      (str)   — leg A ticker
            ticker_b      (str)   — leg B ticker
            shares_a      (float) — signed share count for leg A (+ long, - short)
            shares_b      (float) — signed share count for leg B (+ long, - short)
            price_a       (float) — fill price for leg A in USD
            price_b       (float) — fill price for leg B in USD
            adv_a         (float) — 30-day avg daily volume for leg A (shares)
            adv_b         (float) — 30-day avg daily volume for leg B (shares)
            expected_profit (float) — estimated P&L from spread reversion in USD

    Returns:
        Total round-trip cost in USD, or -1.0 if the trade fails the
        minimum profit-to-cost ratio gate.
    """
    shares_a    = abs(trade["shares_a"])
    shares_b    = abs(trade["shares_b"])
    price_a     = trade["price_a"]
    price_b     = trade["price_b"]
    adv_a       = trade["adv_a"]
    adv_b       = trade["adv_b"]

    # ── Commission ────────────────────────────────────────────────────────────
    commission = (shares_a + shares_b) * CONFIG.commission_per_share

    # ── Slippage ──────────────────────────────────────────────────────────────
    slippage_bps_a = (
        CONFIG.slippage_bps_liquid
        if adv_a > CONFIG.liquidity_threshold_adv
        else CONFIG.slippage_bps_medium
    )
    slippage_bps_b = (
        CONFIG.slippage_bps_liquid
        if adv_b > CONFIG.liquidity_threshold_adv
        else CONFIG.slippage_bps_medium
    )

    slippage = (
        shares_a * price_a * slippage_bps_a / 10_000
        + shares_b * price_b * slippage_bps_b / 10_000
    )

    # ── Bid-ask spread ────────────────────────────────────────────────────────
    bid_ask_bps_a = (
        CONFIG.bid_ask_bps_high_price
        if price_a > CONFIG.bid_ask_price_threshold
        else CONFIG.bid_ask_bps_low_price
    )
    bid_ask_bps_b = (
        CONFIG.bid_ask_bps_high_price
        if price_b > CONFIG.bid_ask_price_threshold
        else CONFIG.bid_ask_bps_low_price
    )

    bid_ask = (
        shares_a * price_a * bid_ask_bps_a / 10_000
        + shares_b * price_b * bid_ask_bps_b / 10_000
    )

    total_cost = commission + slippage + bid_ask

    logger.debug(
        "Cost breakdown %s/%s — commission=$%.2f, slippage=$%.2f, "
        "bid_ask=$%.2f, total=$%.2f",
        trade["ticker_a"], trade["ticker_b"],
        commission, slippage, bid_ask, total_cost,
    )

    # ── Profit-to-cost gate ───────────────────────────────────────────────────
    expected_profit = trade.get("expected_profit", float("inf"))
    round_trip_cost = total_cost * 2  # entry + exit

    if expected_profit < CONFIG.min_profit_to_cost_ratio * round_trip_cost:
        logger.info(
            "SKIPPING %s/%s — expected profit $%.2f < %.1f× round-trip cost $%.2f",
            trade["ticker_a"], trade["ticker_b"],
            expected_profit, CONFIG.min_profit_to_cost_ratio, round_trip_cost,
        )
        return -1.0

    return total_cost


def accrue_borrow_cost(
    short_shares: float,
    price: float,
    holding_days: int = 1,
) -> float:
    """
    Compute the short borrow cost accrued over a holding period.

    Applies CONFIG.short_borrow_annual as a flat annualized rate.
    Call once per trading day per open position with a short leg.

    Args:
        short_shares: Absolute number of shares held short.
        price:        Current market price of the shorted stock in USD.
        holding_days: Number of trading days to accrue (default 1 for daily).

    Returns:
        Borrow cost in USD for the given holding period.
    """
    daily_borrow_rate = CONFIG.short_borrow_annual / 252
    return short_shares * price * daily_borrow_rate * holding_days