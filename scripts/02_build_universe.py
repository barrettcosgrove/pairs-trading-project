# Runs the data cleaning pipeline and applies universe filters to construct data/processed/universe_history.parquet. Re-run after any changes to filter thresholds in config.py.
"""
scripts/02_build_universe.py — Universe Builder

Orchestrates the full data cleaning and universe construction pipeline.
All logic lives in src/data/clean.py and src/universe/filter.py — this
script is a thin wrapper that handles I/O, logging, and orchestration.

Must be run after scripts/01_fetch_data.py has written data/raw/.
Re-run whenever filter thresholds in src/config.py change — delete
data/processed/ first to avoid stale outputs.

Usage:
    uv run python scripts/02_build_universe.py

Outputs (written to data/processed/):
    returns.parquet          — Daily log returns, no NaN, tickers passing QC
    universe_history.parquet — Monthly reconstitution decisions (passed_filters)

Outputs (written to outputs/):
    data_quality_report.txt  — Dropped tickers and return outlier log

Next step: Run scripts/03_run_backtest.py
"""

import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make src/ importable when running from the project root
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clean import clean_prices, write_data_quality_report
from src.data.load import load_prices, load_returns
from src.universe.filter import build_universe_history

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure console logging with timestamps, mirroring 01_fetch_data.py."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    start_time: float,
    date_range: tuple[date, date],
    n_recon_dates: int,
    avg_tickers_per_month: float,
    n_dropped: int,
    quality_report_path: Path,
) -> None:
    """
    Print a human-readable summary of what the script produced.

    Args:
        start_time: Unix timestamp from time.time() at script start.
        date_range: (start_date, end_date) tuple covering the returns data.
        n_recon_dates: Number of monthly reconstitution dates evaluated.
        avg_tickers_per_month: Mean number of passing tickers per month.
        n_dropped: Number of tickers dropped during price cleaning.
        quality_report_path: Path to the written data quality report.
    """
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("  BUILD UNIVERSE COMPLETE")
    print("=" * 60)
    print(f"  Elapsed time        : {elapsed:.1f}s")
    print(f"  Date range          : {date_range[0]} → {date_range[1]}")
    print(f"  Reconstitution dates: {n_recon_dates} monthly")
    print(f"  Avg tickers / month : {avg_tickers_per_month:.1f}")
    print(f"  Tickers dropped (QC): {n_dropped}")
    print()

    files = {
        PROCESSED_DIR / "returns.parquet": "Daily log returns",
        PROCESSED_DIR / "universe_history.parquet": "Monthly universe decisions",
    }
    for path, description in files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK]  {path.name:<35} {size_kb:>7.1f} KB  —  {description}")
        else:
            print(f"  [!!]  {path.name:<35} MISSING         —  {description}")

    print()
    print(f"  Data quality report : {quality_report_path}")
    print("  Review it for dropped tickers and return outliers before")
    print("  running the backtest.")
    print()
    print("  Next step:")
    print("    uv run python scripts/03_run_backtest.py")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    print()
    print("=" * 60)
    print("  ARQ PAIRS TRADING — BUILD UNIVERSE")
    print("=" * 60)
    print(f"  Raw data    : {RAW_DIR.resolve()}")
    print(f"  Processed   : {PROCESSED_DIR.resolve()}")
    print(f"  Reports     : {OUTPUTS_DIR.resolve()}")
    print("=" * 60)
    print()

    start_time = time.time()

    # Check prerequisite
    prices_raw_path = RAW_DIR / "prices.parquet"
    if not prices_raw_path.exists():
        logger.error(
            "data/raw/prices.parquet not found. "
            "Run scripts/01_fetch_data.py first."
        )
        sys.exit(1)

    regime_path = RAW_DIR / "regime.parquet"
    if not regime_path.exists():
        logger.error(
            "data/raw/regime.parquet not found. "
            "Run scripts/01_fetch_data.py first."
        )
        sys.exit(1)

    try:
        # -----------------------------------------------------------------
        # Step 1 — Load raw prices directly (script is the caller;
        #           direct pd.read_parquet is allowed here)
        # -----------------------------------------------------------------
        logger.info("Loading raw prices from %s", prices_raw_path)
        prices_raw = pd.read_parquet(prices_raw_path)
        logger.info(
            "Raw prices: %d tickers, %d rows",
            prices_raw["ticker"].nunique(),
            len(prices_raw),
        )

        # -----------------------------------------------------------------
        # Step 2 — Clean prices → log returns
        # -----------------------------------------------------------------
        logger.info("Running clean_prices()...")
        returns, dropped_tickers, flagged_returns = clean_prices(prices_raw)

        # -----------------------------------------------------------------
        # Step 3 — Write log returns to data/processed/returns.parquet
        # -----------------------------------------------------------------
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        returns_path = PROCESSED_DIR / "returns.parquet"
        returns.to_parquet(returns_path, index=False)
        logger.info(
            "Returns written to %s — %d tickers, %d rows",
            returns_path,
            returns["ticker"].nunique(),
            len(returns),
        )

        # -----------------------------------------------------------------
        # Step 4 — Write data quality report
        # -----------------------------------------------------------------
        quality_report_path = OUTPUTS_DIR / "data_quality_report.txt"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        write_data_quality_report(dropped_tickers, flagged_returns, quality_report_path)

        # -----------------------------------------------------------------
        # Step 5 — Re-load data via load.py for the filter step
        # Returns are loaded back via load.py to exercise the same typed
        # interface that all downstream modules use.
        # -----------------------------------------------------------------
        start_date = returns["date"].min().date()
        end_date = returns["date"].max().date()

        logger.info("Loading cleaned returns and prices via load.py...")
        loaded_returns = load_returns(start_date, end_date)
        prices = load_prices(start_date, end_date)

        # -----------------------------------------------------------------
        # Step 6 — Compute SPY log returns from regime data
        # regime.parquet is read directly here (no load_spy() in load.py).
        # SPY is a regime/benchmark ticker, not a candidate ticker.
        # -----------------------------------------------------------------
        logger.info("Computing SPY log returns from %s", regime_path)
        regime = pd.read_parquet(regime_path)
        regime["date"] = pd.to_datetime(regime["date"])
        regime = regime[
            (regime["date"] >= pd.Timestamp(start_date))
            & (regime["date"] <= pd.Timestamp(end_date))
        ].sort_values("date")

        spy_prices = regime.set_index("date")["spy"]
        # log(spy_t / spy_{t-1}) — first row becomes NaN and is dropped
        spy_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()
        spy_returns.name = "spy_log_return"
        logger.info(
            "SPY log returns: %d observations (%s → %s)",
            len(spy_returns),
            spy_returns.index.min().date(),
            spy_returns.index.max().date(),
        )

        # -----------------------------------------------------------------
        # Step 7 — Generate monthly reconstitution dates
        # BMS = Business Month Start: first trading day of each month.
        # Using the returns date range so reconstitution is only evaluated
        # where we actually have return data.
        # -----------------------------------------------------------------
        recon_timestamps = pd.date_range(
            start=loaded_returns["date"].min(),
            end=loaded_returns["date"].max(),
            freq="BMS",
        )
        recon_dates: list[date] = [ts.date() for ts in recon_timestamps]
        logger.info(
            "Generated %d monthly reconstitution dates (%s → %s)",
            len(recon_dates),
            recon_dates[0],
            recon_dates[-1],
        )

        # -----------------------------------------------------------------
        # Step 8 — Build universe history
        # -----------------------------------------------------------------
        logger.info("Running build_universe_history()...")
        universe_history = build_universe_history(
            prices=prices,
            returns=loaded_returns,
            spy_returns=spy_returns,
            reconstitution_dates=recon_dates,
        )

    except Exception as exc:
        logger.error("Build universe failed: %s", exc, exc_info=True)
        sys.exit(1)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    passing = universe_history[universe_history["passed_filters"]]
    avg_tickers_per_month = (
        passing.groupby("date").size().mean() if not passing.empty else 0.0
    )

    print_summary(
        start_time=start_time,
        date_range=(start_date, end_date),
        n_recon_dates=len(recon_dates),
        avg_tickers_per_month=avg_tickers_per_month,
        n_dropped=len(dropped_tickers),
        quality_report_path=quality_report_path,
    )


if __name__ == "__main__":
    main()
