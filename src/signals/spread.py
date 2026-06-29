# Computes spread = log(P_A) - β × log(P_B) at as_of and formation z-score
# (spread_today − μ_F) / max(σ_F, CONFIG.min_formation_spread_std).

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import CONFIG, StrategyConfig

logger = logging.getLogger(__name__)


def compute(
    ticker_a: str,
    ticker_b: str,
    beta: float,
    mean: float,
    std: float,
    prices: pd.DataFrame,
    as_of: date,
    config: StrategyConfig | None = None,
) -> tuple[float, float]:
    """
    Compute today's log spread and formation z-score vs locked μ_F, σ_F.

    Uses formation hedge ratio β_F with today's log prices through ``as_of`` only.
    z = (s_t − μ_F) / max(σ_F, min_formation_spread_std) when σ_F is valid.

    Args:
        ticker_a: Ticker symbol for leg A (dependent in the OLS formation fit).
        ticker_b: Ticker symbol for leg B.
        beta: Locked hedge ratio β_F from the formation window.
        mean: Locked formation spread mean μ_F.
        std: Locked formation spread standard deviation σ_F.
        prices: DataFrame with columns [date, ticker, adj_close].
        as_of: Observation date; only rows with date <= as_of are used.
        config: Strategy parameters; defaults to ``CONFIG`` when None.

    Returns:
        Tuple ``(spread_value, z_score)``. ``spread_value`` is
        log(P_A) − β_F log(P_B) on the last aligned row on or before ``as_of``.
        ``z_score`` is the formation z, or NaN if inputs are invalid or prices
        are missing.
    """
    cfg = config or CONFIG
    prices_to_date = prices[prices["date"] <= pd.Timestamp(as_of)]
    pivot = prices_to_date.pivot(index="date", columns="ticker", values="adj_close")

    if ticker_a not in pivot.columns or ticker_b not in pivot.columns:
        logger.warning(
            "Missing price data for %s or %s as of %s", ticker_a, ticker_b, as_of
        )
        return float("nan"), float("nan")

    aligned = pivot[[ticker_a, ticker_b]].dropna()
    if aligned.empty:
        logger.warning(
            "No overlapping prices for spread %s/%s as of %s", ticker_a, ticker_b, as_of
        )
        return float("nan"), float("nan")

    last_row = aligned.iloc[-1]
    log_a = float(np.log(last_row[ticker_a]))
    log_b = float(np.log(last_row[ticker_b]))
    spread_today = log_a - beta * log_b

    if beta != beta or mean != mean or std != std:
        logger.warning(
            "Non-finite formation stats for %s/%s on %s — returning NaN z",
            ticker_a,
            ticker_b,
            as_of,
        )
        return float(spread_today), float("nan")

    sigma_f = float(std)
    if sigma_f <= 0.0 or sigma_f != sigma_f:
        logger.warning(
            "Invalid formation std for spread %s/%s on %s — returning NaN z",
            ticker_a,
            ticker_b,
            as_of,
        )
        return float(spread_today), float("nan")

    denom = max(sigma_f, cfg.min_formation_spread_std)
    z_score = (float(spread_today) - float(mean)) / denom

    logger.debug(
        "Spread %s/%s as of %s — value=%.4f, formation z=%.2f",
        ticker_a,
        ticker_b,
        as_of,
        spread_today,
        z_score,
    )

    return float(spread_today), float(z_score)
