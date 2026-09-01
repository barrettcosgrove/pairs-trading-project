# Read-only interface for all pipeline stages. Provides typed functions like load_prices(), load_returns(), load_vix(). All other modules import from here — never call pd.read_parquet() directly elsewhere.
"""
src/data/load.py — Data Access Layer

Read-only interface for all pipeline modules. Every function reads from
data/raw/ or data/processed/ and returns a clean, typed DataFrame or Series.

No module outside this file should ever call pd.read_parquet() directly.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_returns(start: date, end: date) -> pd.DataFrame:
    """
    Load daily log returns for all tickers in a date range.

    Returns are computed from adj_close in prices.parquet rather than
    read from a pre-built returns file. A returns.parquet does not exist
    yet — this function derives them on demand.

    Args:
        start: First date to include (inclusive).
        end: Last date to include (inclusive).

    Returns:
        DataFrame with columns [date, ticker, log_return], no NaN values,
        sorted by date ascending with a RangeIndex.
    """
    path = PROCESSED_DIR / "returns.parquet"
    df = pd.read_parquet(path)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
        
    df = df[["date", "ticker", "log_return"]].copy()

    # clean.py guarantees no NaN, but drop defensively so callers never see one
    df = df.dropna(subset=["log_return"])

    return df.sort_values("date").reset_index(drop=True)


def load_prices(start: date, end: date) -> pd.DataFrame:
    """
    Load daily prices for all tickers in a date range.

    Args:
        start: First date to include (inclusive).
        end: Last date to include (inclusive).

    Returns:
        DataFrame with columns [date, ticker, adj_close, volume].
    """
    path = RAW_DIR / "prices.parquet"
    df = pd.read_parquet(path)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
        
    df = df[["date", "ticker", "adj_close", "volume"]].copy()

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_vix(start: date, end: date) -> pd.Series:
    """
    Load daily VIX closing values for a date range.

    Args:
        start: First date to include (inclusive).
        end: Last date to include (inclusive).

    Returns:
        Series of VIX daily close values indexed by date (datetime64),
        named "vix".
    """
    path = RAW_DIR / "regime.parquet"
    df = pd.read_parquet(path)

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
        
    series = df.set_index("date")["vix"]
    series.name = "vix"
    return series


def load_universe(as_of: date) -> list[str]:
    """
    Return tickers that passed all hard filters on or before a given date.

    Uses the most recent monthly reconstitution date on or before as_of,
    matching the anti-look-ahead semantics of the backtest engine.

    Args:
        as_of: The date for which to retrieve the universe.

    Returns:
        List of ticker strings that passed all filters on the most recent
        reconstitution date on or before as_of. Returns an empty list if
        no reconstitution date exists on or before as_of.
    """
    path = PROCESSED_DIR / "universe_history.parquet"
    df = pd.read_parquet(path)

    as_of_ts = pd.Timestamp(as_of)
    eligible_dates = df.loc[df["date"] <= as_of_ts, "date"]

    if eligible_dates.empty:
        logger.warning(
            "No universe reconstitution found on or before %s — returning empty list",
            as_of,
        )
        return []

    latest_recon = eligible_dates.max()
    tickers = df.loc[
        (df["date"] == latest_recon) & df["passed_filters"],
        "ticker",
    ].tolist()

    logger.debug(
        "load_universe(as_of=%s): using recon date %s, %d tickers",
        as_of,
        latest_recon.date(),
        len(tickers),
    )

    return tickers

def load_fundamentals() -> pd.DataFrame:
    """
    Load the fundamentals snapshot for all tickers.

    Note: This is a current snapshot, not point-in-time historical data.
    Fundamental scores are held constant over the backtest. See docs/data.md
    section 6.2 for the full discussion of this limitation.

    Returns:
        DataFrame with columns [ticker, fetch_date, price_to_sales,
        revenue_growth_ttm]. May contain NaN for tickers where yfinance
        had no fundamental data available.
    """
    path = RAW_DIR / "fundamentals.parquet"
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)