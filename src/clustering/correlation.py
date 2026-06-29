# Computes pairwise Pearson correlations from daily log returns over the 120-day clustering window. Converts to a distance matrix (D = 1 - ρ). Saves monthly snapshots to data/processed/correlation_matrices/.
"""
src/clustering/correlation.py - Correlation Distance Matrix Builder

Builds the pairwise ticker distance matrix used by the clustering layer.
The module computes Pearson correlations from trailing log returns, converts
them to distances via D = 1 - rho, and can persist monthly snapshots for
downstream reuse by kmeans.py.
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.data.load import load_returns

logger = logging.getLogger(__name__)

CORRELATION_OUTPUT_DIR = Path("data/processed/correlation_matrices")
REQUIRED_RETURN_COLUMNS = {"date", "ticker", "log_return"}

# This is the main function for distance matrix, composed of many smaller helper functions
def build_distance_matrix(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Build a ticker-by-ticker distance matrix from trailing log returns.

    Args:
        returns: Long-form return data with columns [date, ticker, log_return].
            The data must include only observations available on or before the
            intended as_of date.
        window: Number of trailing trading days to use for the correlation
            calculation. The clustering workflow should pass
            CONFIG.clustering_window.

    Returns:
        NxN DataFrame whose index and columns are ticker strings and whose
        values are distances in [0, 2] computed as 1 - correlation.
    """
    
    # Checks that {date, ticker, log_return} are all present
    _validate_returns_columns(returns)

    # Finds last 'window unique trading dates
    # returns filtered long-form DataFrame
    trailing_returns = _select_trailing_window(returns, window)

    # Pivots to wide [date x ticker] --> better for .corr() input
    # Drops any tickers with NaN
    returns_wide = _pivot_returns_wide(trailing_returns)


    correlation_matrix = _compute_correlation_matrix(returns_wide)
    distance_matrix = _correlation_to_distance(correlation_matrix)

    logger.info(
        "Built distance matrix for %d tickers using %d trailing trading days",
        distance_matrix.shape[0],
        window,
    )

    return distance_matrix


def build_and_save_distance_matrix(
    as_of: date,
    window: int = CONFIG.clustering_window,
) -> pd.DataFrame:
    """
    Load trailing returns through an as_of date, build the distance matrix,
    and persist the monthly snapshot to disk.

    Args:
        as_of: Snapshot date. Only return observations on or before this date
            are loaded, preventing look-ahead bias.
        window: Number of trailing trading days to use when computing the
            correlation matrix. Defaults to CONFIG.clustering_window.

    Returns:
        NxN distance matrix for the requested snapshot date. The same matrix is
        also written to data/processed/correlation_matrices/YYYY-MM.parquet.
    """
    returns = _load_returns_for_snapshot(as_of, window)
    distance_matrix = build_distance_matrix(returns=returns, window=window)
    output_path = save_distance_matrix(distance_matrix=distance_matrix, as_of=as_of)

    logger.info("Saved distance matrix snapshot for %s to %s", as_of, output_path)

    return distance_matrix


def save_distance_matrix(distance_matrix: pd.DataFrame, as_of: date) -> Path:
    """
    Write a distance matrix snapshot to the monthly clustering cache.

    Args:
        distance_matrix: Ticker-by-ticker distance matrix produced by
            build_distance_matrix().
        as_of: Snapshot date used to derive the YYYY-MM output filename.

    Returns:
        Path to the written parquet file.
    """
    output_path = _build_snapshot_path(as_of)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distance_matrix.to_parquet(output_path)
    return output_path


def _validate_returns_columns(returns: pd.DataFrame) -> None:
    """
    Validate that the returns DataFrame matches the clustering input contract.

    Args:
        returns: Candidate return data passed to build_distance_matrix().

    Returns:
        None. Raises ValueError if required columns are missing.
    """
    missing_columns = REQUIRED_RETURN_COLUMNS.difference(returns.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Returns DataFrame is missing required columns: {missing_list}")


def _select_trailing_window(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Restrict long-form returns to the most recent trailing trading window.

    Args:
        returns: Long-form return data with [date, ticker, log_return] columns.
        window: Number of trailing trading dates required.

    Returns:
        Filtered long-form DataFrame containing only rows whose dates fall
        within the most recent trailing window.
    """
    if window != CONFIG.clustering_window:
        logger.debug(
            "build_distance_matrix received window=%d; clustering default is %d",
            window,
            CONFIG.clustering_window,
        )

    if window <= 0:
        raise ValueError("window must be a positive integer")

    returns_sorted = returns.copy()
    returns_sorted["date"] = pd.to_datetime(returns_sorted["date"])
    returns_sorted = returns_sorted.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = returns_sorted["date"].drop_duplicates().sort_values()
    if len(unique_dates) < window:
        raise ValueError(
            f"Need at least {window} trading dates to build the distance matrix; "
            f"received {len(unique_dates)}"
        )

    trailing_dates = unique_dates.iloc[-window:]
    trailing_returns = returns_sorted[returns_sorted["date"].isin(trailing_dates)].copy()

    return trailing_returns


def _pivot_returns_wide(returns_window: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape long-form returns into a wide date-by-ticker matrix.

    Args:
        returns_window: Trailing-window return data with columns
            [date, ticker, log_return].

    Returns:
        Wide DataFrame indexed by date with one numeric log_return column
        per ticker. Tickers with missing values in the trailing window are
        dropped and logged.
    """
    returns_wide = returns_window.pivot(
        index="date", 
        columns="ticker", 
        values="log_return"
    )
    
    returns_wide = returns_wide.sort_index().sort_index(axis=1)

    # Drop tickers with incomplete return history in the trailing window
    # instead of crashing the whole snapshot build.
    if returns_wide.isna().any().any():
        missing_tickers = returns_wide.columns[returns_wide.isna().any()].tolist()

        logger.warning(
            "Dropping %d ticker(s) with incomplete return history in the "
            "trailing window: %s",
            len(missing_tickers),
            ", ".join(missing_tickers),
        )

        returns_wide = returns_wide.drop(columns=missing_tickers)

    # Added:
    # Fail only if too few tickers remain to form a meaningful matrix.
    if returns_wide.shape[1] < CONFIG.k_min:
        raise ValueError(
            f"Need at least {CONFIG.k_min} tickers after dropping incomplete "
            f"series; received {returns_wide.shape[1]}"
        )

    return returns_wide


def _compute_correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix across ticker return columns.

    Args:
        returns_wide: Wide return matrix indexed by date with ticker columns.

    Returns:
        NxN Pearson correlation matrix whose index and columns are ticker
        strings.
    """
    correlation_matrix = returns_wide.corr(method="pearson")

    if correlation_matrix.isna().any().any():
        invalid_tickers = correlation_matrix.columns[correlation_matrix.isna().any()].tolist()
        raise ValueError(
            "Correlation matrix contains NaN values for tickers: (likely zero-variance "
            "return series — ticker may be halted or illiquid): "
            + ", ".join(invalid_tickers)
        )

    return correlation_matrix


def _correlation_to_distance(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a correlation matrix to a distance matrix via D = 1 - rho.

    Args:
        correlation_matrix: NxN ticker correlation matrix.

    Returns:
        NxN ticker distance matrix with values clipped to [0, 2] and zeros
        on the diagonal.
    """
    bounded_correlation = correlation_matrix.clip(lower=-1.0, upper=1.0)
    distance_matrix = 1.0 - bounded_correlation
    distance_matrix = distance_matrix.clip(lower=0.0, upper=2.0)
    values = distance_matrix.values.copy()
    np.fill_diagonal(values, 0.0)
    dist_matrix = pd.DataFrame(
        values,
        index=distance_matrix.index,
        columns=distance_matrix.columns,
    )

    return dist_matrix


def _load_returns_for_snapshot(as_of: date, window: int) -> pd.DataFrame:
    """
    Load only the return history needed to build a point-in-time snapshot.

    Args:
        as_of: Snapshot date. No observations after this date are loaded.
        window: Number of trailing trading days required by the correlation
            calculation.

    Returns:
        Long-form return DataFrame with columns [date, ticker, log_return]
        covering the requested snapshot horizon.
    """
    end_ts = pd.Timestamp(as_of)
    extra_buffer_days = round(window * CONFIG.extbuffer_percent) # 40% extra to survive holidays/gaps
    start_ts = end_ts - pd.offsets.BDay(window + extra_buffer_days)
    returns = load_returns(start=start_ts.date(), end=as_of)

    logger.info(
        "Loaded %d return rows for correlation snapshot through %s",
        len(returns),
        as_of,
    )

    return returns


def _build_snapshot_path(as_of: date) -> Path:
    """
    Build the monthly parquet path for a correlation distance snapshot.

    Args:
        as_of: Snapshot date used to derive the YYYY-MM filename.

    Returns:
        Output path under data/processed/correlation_matrices/.
    """
    timestamp = pd.Timestamp(as_of)
    return _build_snapshot_path_from_parts(timestamp.year, timestamp.month)


def _build_snapshot_path_from_parts(year: int, month: int) -> Path:
    """
    Build the monthly parquet path for a correlation distance snapshot.

    Args:
        year: Calendar year for the snapshot.
        month: Calendar month for the snapshot.

    Returns:
        Output path under data/processed/correlation_matrices/.
    """
    filename = f"{year:04d}-{month:02d}.parquet"
    return CORRELATION_OUTPUT_DIR / filename


def snapshot_exists(year: int, month: int) -> bool:
    """
    Check whether a monthly distance matrix snapshot already exists.

    Args:
        year: Calendar year for the snapshot.
        month: Calendar month for the snapshot.

    Returns:
        True if the snapshot parquet file exists, otherwise False.
    """
    snapshot_path = _build_snapshot_path_from_parts(year, month)
    return snapshot_path.exists()


def load_distance_matrix_snapshot(year: int, month: int) -> pd.DataFrame:
    """
    Load a cached monthly distance matrix snapshot from disk.

    Args:
        year: Calendar year for the snapshot.
        month: Calendar month for the snapshot.

    Returns:
        NxN distance matrix whose index and columns are ticker strings.

    Raises:
        FileNotFoundError: If the requested snapshot does not exist.
    """
    snapshot_path = _build_snapshot_path_from_parts(year, month)
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"No correlation snapshot found for {year:04d}-{month:02d} at "
            f"{snapshot_path}"
        )

    # pd.read_parquet() is permitted here as a deliberate exception to the load.py
    # rule. Correlation snapshots are clustering-internal state: this module is the
    # sole writer and reader, the schema is fixed (NxN ticker DataFrame), and no
    # date filtering or column validation is needed. Shared market data still goes
    # through src/data/load.py.
    distance_matrix = pd.read_parquet(snapshot_path)

    logger.info(
        "Loaded distance matrix snapshot for %04d-%02d from %s",
        year,
        month,
        snapshot_path,
    )

    return distance_matrix

