"""
src/data/clean.py — Price Data Cleaner

Validates raw price data, forward-fills single-day gaps, drops tickers with
too many missing days, and computes daily log returns. Called exclusively by
scripts/02_build_universe.py — never run directly.

Does not read or write parquet files. All I/O is handled by the caller.

Outputs (returned, not written):
    returns         — DataFrame[ticker, date, log_return], no NaN
    dropped_tickers — dict mapping ticker → reason for every dropped ticker
    flagged_returns — DataFrame of |log_return| > 0.50 rows for quality report
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)


def clean_prices(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """
    Validate and clean raw OHLCV prices, then compute daily log returns.

    Runs these checks in order:
      1. Sort by [ticker, date].
      2. Forward-fill adj_close for exactly 1 consecutive missing trading day;
         log a warning for any gap of 2 or more consecutive days (not filled).
      3. Drop tickers where any trailing CONFIG.data_quality_window-day window
         contains more than CONFIG.max_missing_days missing adj_close values.
      4. Compute log returns: log(adj_close_t / adj_close_{t-1}) per ticker.
      5. Drop the first row per ticker (NaN from differencing).
      6. Collect rows where |log_return| > 0.50 into flagged_returns and log
         a warning for each — outliers are kept, not dropped.

    Args:
        prices: Raw price DataFrame with columns
            [ticker, date, adj_close, volume]. Produced by fetch.py and read
            from data/raw/prices.parquet by the caller.

    Returns:
        Tuple of:
          - returns (pd.DataFrame): columns [ticker, date, log_return],
            no NaN values, sorted by date ascending.
          - dropped_tickers (dict[str, str]): ticker → reason string for
            every ticker excluded from returns.
          - flagged_returns (pd.DataFrame): subset of returns where
            |log_return| > 0.50, for inclusion in the quality report.
    """
    # -------------------------------------------------------------------------
    # Step 1 — Sort
    # -------------------------------------------------------------------------
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    dropped: dict[str, str] = {}

    # Pivot to wide format so every ticker shares a common date index.
    # Reindexing to all observed dates introduces NaN for missing trading days.
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    wide.index = pd.DatetimeIndex(wide.index)
    wide = wide.sort_index()
    wide.columns.name = None  # remove "ticker" label from column axis

    # -------------------------------------------------------------------------
    # Step 2 — Forward-fill single-day gaps; warn on extended gaps
    # -------------------------------------------------------------------------
    for ticker in wide.columns:
        series = wide[ticker]
        is_null = series.isna()

        if not is_null.any():
            continue

        # Label each consecutive run of identical boolean values.
        # run_id increments at every True→False or False→True transition.
        run_id = (is_null != is_null.shift()).cumsum()

        # run_lengths[i] = number of NaN values in the consecutive run at
        # position i. Non-NaN runs get 0; NaN runs get their length.
        run_lengths = is_null.groupby(run_id).transform("sum")

        # Warn on extended gaps (2+ consecutive missing days) — do not fill.
        extended = is_null & (run_lengths >= 2)
        if extended.any():
            logger.warning(
                "%s has %d trading day(s) inside extended gaps (2+ consecutive) "
                "— these rows will not be forward-filled",
                ticker,
                int(extended.sum()),
            )

        # Forward-fill only positions that are single-day gaps.
        # mask(cond, other): where cond is True, substitute other's value.
        single_gaps = is_null & (run_lengths == 1)
        if single_gaps.any():
            wide[ticker] = series.mask(single_gaps, series.ffill())

    # -------------------------------------------------------------------------
    # Step 3 — Drop tickers with too many missing days in any trailing window
    # -------------------------------------------------------------------------
    for ticker in list(wide.columns):
        missing = wide[ticker].isna()

        # Rolling sum of NaN indicators over trailing CONFIG.data_quality_window
        # rows. min_periods=1 so short histories are still evaluated.
        rolling_missing = missing.rolling(
            window=CONFIG.data_quality_window, min_periods=1
        ).sum()

        if rolling_missing.max() > CONFIG.max_missing_days:
            dropped[ticker] = (
                f"too many missing days in trailing "
                f"{CONFIG.data_quality_window}-day window"
            )
            logger.warning(
                "Dropping %s: %d missing days exceeded threshold of %d",
                ticker,
                int(rolling_missing.max()),
                CONFIG.max_missing_days,
            )

    # Remove dropped tickers from the wide matrix before computing returns.
    keep = [t for t in wide.columns if t not in dropped]
    wide = wide[keep]

    # -------------------------------------------------------------------------
    # Step 4 — Compute log returns per ticker via groupby + transform
    # -------------------------------------------------------------------------
    # Melt wide → long so groupby operates on a single "ticker" key column.
    long = (
        wide.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="adj_close")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    # transform applies shift(1) within each ticker's group, preventing
    # the last row of ticker A from being the denominator for ticker B's
    # first return when the DataFrame is sorted by ticker then date.
    long["log_return"] = long.groupby("ticker")["adj_close"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # -------------------------------------------------------------------------
    # Step 5 — Drop first row per ticker (NaN from differencing) and any
    #           remaining NaN produced by extended-gap rows
    # -------------------------------------------------------------------------
    long = long.dropna(subset=["log_return"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 6 — Collect flagged return outliers (|log_return| > 0.50)
    # -------------------------------------------------------------------------
    outlier_mask = long["log_return"].abs() > 0.50
    flagged_returns = long.loc[outlier_mask, ["ticker", "date", "log_return"]].copy()

    for _, row in flagged_returns.iterrows():
        date_str = (
            row["date"].strftime("%Y-%m-%d")
            if hasattr(row["date"], "strftime")
            else str(row["date"])
        )
        logger.warning(
            "Return outlier: %s on %s — log_return=%.4f (kept)",
            row["ticker"],
            date_str,
            row["log_return"],
        )

    # -------------------------------------------------------------------------
    # Step 7 — Return
    # -------------------------------------------------------------------------
    returns = (
        long[["ticker", "date", "log_return"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info(
        "clean_prices complete — %d tickers, %d return rows, "
        "%d dropped, %d outliers flagged",
        returns["ticker"].nunique(),
        len(returns),
        len(dropped),
        len(flagged_returns),
    )

    return returns, dropped, flagged_returns


def write_data_quality_report(
    dropped_tickers: dict[str, str],
    flagged_returns: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write a plain-text data quality report summarising cleaning results.

    Args:
        dropped_tickers: Mapping of ticker → reason for every ticker dropped
            by clean_prices.
        flagged_returns: DataFrame of return outlier rows with columns
            [ticker, date, log_return], as returned by clean_prices.
        out_path: Destination path for the report file. Parent directories
            are created if they do not exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "ARQ Pairs Trading — Data Quality Report",
        f"Generated: {timestamp}",
        "=" * 60,
        "",
        f"Dropped Tickers ({len(dropped_tickers)} total)",
        "-" * 40,
    ]

    if dropped_tickers:
        for ticker in sorted(dropped_tickers):
            lines.append(f"  {ticker}: {dropped_tickers[ticker]}")
    else:
        lines.append("  None")

    lines += [
        "",
        f"Return Outliers — |log_return| > 0.50 ({len(flagged_returns)} total)",
        "-" * 40,
    ]

    if not flagged_returns.empty:
        for _, row in flagged_returns.sort_values(["date", "ticker"]).iterrows():
            date_str = (
                row["date"].strftime("%Y-%m-%d")
                if hasattr(row["date"], "strftime")
                else str(row["date"])
            )
            lines.append(
                f"  {row['ticker']} on {date_str}: log_return={row['log_return']:.4f}"
            )
    else:
        lines.append("  None")

    lines += [
        "",
        "Review complete — check flagged items before running backtest",
    ]

    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Data quality report written to %s", out_path)
