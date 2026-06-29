# Generates a single trading signal for a pair on a given date using formation
# z-score vs CONFIG entry/exit thresholds. Returns one of six string literals
# consumed by the backtest engine.

import logging
from datetime import date

import pandas as pd

from src.config import CONFIG, StrategyConfig
from src.signals.spread import compute as compute_spread

logger = logging.getLogger(__name__)

# Valid signal literals — engine matches on these strings exactly
LONG_SPREAD = "LONG_SPREAD"
SHORT_SPREAD = "SHORT_SPREAD"
TAKE_PROFIT = "TAKE_PROFIT"
STOP_LOSS = "STOP_LOSS"
TIME_STOP = "TIME_STOP"
HOLD = "HOLD"


def _leg_momentum_return(
    prices: pd.DataFrame,
    ticker: str,
    as_of: date,
    window: int,
) -> float | None:
    """
    Compute cumulative return over ``window`` trading days for one ticker.

    Uses only rows with date on or before ``as_of``, sorted by date.

    Args:
        prices: Long-form prices with columns [date, ticker, adj_close].
        ticker: Ticker symbol.
        as_of: End date for the window (inclusive).
        window: Number of trading days in the return horizon.

    Returns:
        Fractional return from first to last close in the window, or None if
        data are insufficient or the first close is non-positive.
    """
    sub = prices[
        (prices["date"] <= pd.Timestamp(as_of)) & (prices["ticker"] == ticker)
    ].sort_values("date")
    if len(sub) < window + 1:
        return None
    tail = sub.tail(window + 1)
    first = float(tail.iloc[0]["adj_close"])
    last = float(tail.iloc[-1]["adj_close"])
    if first <= 0.0:
        return None
    return (last - first) / first


def _is_momentum_breakout(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
    config: StrategyConfig | None = None,
) -> bool:
    """
    Return True if either leg moved beyond CONFIG momentum threshold.

    Args:
        ticker_a: Ticker for leg A.
        ticker_b: Ticker for leg B.
        prices: Long-form prices with columns [date, ticker, adj_close].
        as_of: Signal date; only data on or before this date are used.
        config: Strategy parameters; defaults to ``CONFIG`` when None.

    Returns:
        True if either leg's ``window``-day return exceeds ``momentum_threshold``
        in absolute value; False otherwise.
    """
    cfg = config or CONFIG
    window = cfg.momentum_window
    threshold = cfg.momentum_threshold

    ret_a = _leg_momentum_return(prices, ticker_a, as_of, window)
    ret_b = _leg_momentum_return(prices, ticker_b, as_of, window)
    if ret_a is None or ret_b is None:
        return False

    if abs(ret_a) > threshold or abs(ret_b) > threshold:
        logger.info(
            "MOMENTUM BLOCK: %s (%.1f%%) vs %s (%.1f%%)",
            ticker_a,
            ret_a * 100.0,
            ticker_b,
            ret_b * 100.0,
        )
        return True

    return False


def get_signal(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
    beta_formation: float,
    mean_formation: float,
    std_formation: float,
    expected_halflife: float,
    days_open: int,
    current_position: str | None = None,
    config: StrategyConfig | None = None,
) -> str:
    """
    Generate a trading signal for a pair on a given date.

    Computes formation z-score of the spread and maps it to entry/exit
    strings using CONFIG thresholds. Exits are evaluated before entries when
    a position is open.

    Args:
        ticker_a: Ticker symbol for leg A.
        ticker_b: Ticker symbol for leg B.
        prices: DataFrame with columns [date, ticker, adj_close].
        as_of: Signal date. Only data on or before this date is used.
        beta_formation: Locked beta from the formation period.
        mean_formation: Locked spread mean from the formation period.
        std_formation: Locked spread std from the formation period.
        expected_halflife: Half-life estimate from scoring (informational; time
            stop uses CONFIG.time_stop_days).
        days_open: Trading days the position has been open; use 0 when flat.
        current_position: ``LONG_SPREAD``, ``SHORT_SPREAD``, or None when flat.
        config: Strategy parameters; defaults to ``CONFIG`` when None.

    Returns:
        One of: ``LONG_SPREAD``, ``SHORT_SPREAD``, ``TAKE_PROFIT``,
        ``STOP_LOSS``, ``TIME_STOP``, ``HOLD``.
    """
    cfg = config or CONFIG

    if beta_formation != beta_formation:
        logger.warning(
            "NaN hedge ratio for %s/%s on %s — returning HOLD", ticker_a, ticker_b, as_of
        )
        return HOLD

    _, z_score = compute_spread(
        ticker_a,
        ticker_b,
        beta_formation,
        mean_formation,
        std_formation,
        prices,
        as_of,
        config=cfg,
    )

    if z_score != z_score:
        logger.warning(
            "NaN z_score for %s/%s on %s — returning HOLD", ticker_a, ticker_b, as_of
        )
        return HOLD

    if current_position is not None:
        if days_open >= cfg.time_stop_days:
            logger.info(
                "TIME_STOP %s/%s — position open %d days (limit %d)",
                ticker_a,
                ticker_b,
                days_open,
                cfg.time_stop_days,
            )
            return TIME_STOP

        if current_position == LONG_SPREAD:
            if z_score <= -cfg.stop_loss_zscore:
                logger.info(
                    "STOP_LOSS %s/%s — z_score=%.2f (threshold: <= %.2f)",
                    ticker_a,
                    ticker_b,
                    z_score,
                    -cfg.stop_loss_zscore,
                )
                return STOP_LOSS
            if z_score >= -cfg.take_profit_zscore:
                logger.info(
                    "TAKE_PROFIT %s/%s — z_score=%.2f (threshold: >= %.2f)",
                    ticker_a,
                    ticker_b,
                    z_score,
                    -cfg.take_profit_zscore,
                )
                return TAKE_PROFIT

        elif current_position == SHORT_SPREAD:
            if z_score >= cfg.stop_loss_zscore:
                logger.info(
                    "STOP_LOSS %s/%s — z_score=%.2f (threshold: >= %.2f)",
                    ticker_a,
                    ticker_b,
                    z_score,
                    cfg.stop_loss_zscore,
                )
                return STOP_LOSS
            if z_score <= cfg.take_profit_zscore:
                logger.info(
                    "TAKE_PROFIT %s/%s — z_score=%.2f (threshold: <= %.2f)",
                    ticker_a,
                    ticker_b,
                    z_score,
                    cfg.take_profit_zscore,
                )
                return TAKE_PROFIT

        return HOLD

    if z_score <= -cfg.entry_zscore:
        if _is_momentum_breakout(ticker_a, ticker_b, prices, as_of, config=cfg):
            return HOLD
        logger.info(
            "LONG_SPREAD %s/%s — z_score=%.2f <= entry threshold %.2f",
            ticker_a,
            ticker_b,
            z_score,
            -cfg.entry_zscore,
        )
        return LONG_SPREAD

    if z_score >= cfg.entry_zscore:
        if _is_momentum_breakout(ticker_a, ticker_b, prices, as_of, config=cfg):
            return HOLD
        logger.info(
            "SHORT_SPREAD %s/%s — z_score=%.2f >= entry threshold %.2f",
            ticker_a,
            ticker_b,
            z_score,
            cfg.entry_zscore,
        )
        return SHORT_SPREAD

    return HOLD
