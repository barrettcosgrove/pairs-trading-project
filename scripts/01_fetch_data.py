# Runs once at project start (and quarterly for refreshes). Calls src/data/fetch.py to download all price, fundamental, and regime data. Prints fetch summary and failed tickers. Expected runtime: 2-5 minutes.

"""
scripts/01_fetch_data.py — Data Fetch Script

Entry point for downloading all raw data from yfinance. Run this once at
project start, and again for quarterly refreshes.

This script orchestrates src/data/fetch.py. It sets up logging, validates
the output directory, runs the fetch, and prints a human-readable summary
of what was written.

Usage:
    uv run python scripts/01_fetch_data.py
    uv run python scripts/01_fetch_data.py --years 3.5
    uv run python scripts/01_fetch_data.py --years 3.5 --retries 5
    uv run python scripts/01_fetch_data.py --stage fundamentals --resume

Outputs (written to data/raw/):
    prices.parquet       — Daily OHLCV for all candidate tickers
    fundamentals.parquet — P/S ratio and revenue growth snapshot
    regime.parquet       — Daily VIX close and SPY adjusted close

Expected runtime: 2-5 minutes depending on network speed.

Next step: Run scripts/02_build_universe.py
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Make sure src/ is importable when running from the project root
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import FETCH_STATUS_PATH, RAW_DIR, fetch_all

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to print to console with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch all raw data for the ARQ pairs trading pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/01_fetch_data.py
  uv run python scripts/01_fetch_data.py --years 4.0
  uv run python scripts/01_fetch_data.py --retries 5
  uv run python scripts/01_fetch_data.py --stage regime
  uv run python scripts/01_fetch_data.py --stage fundamentals --resume
        """,
    )

    parser.add_argument(
        "--years",
        type=float,
        default=3.5,
        help=(
            "Years of price history to fetch. Default 3.5 gives 2.5 years "
            "for training + 6 months OOS + buffer for rolling window warmup. "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help=(
            "Number of retry attempts per ticker on network failure. "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--disable-proxy",
        action="store_true",
        help=(
            "Disable proxy environment variables for this run. "
            "Useful if yfinance fails with CONNECT tunnel 403 errors."
        ),
    )

    parser.add_argument(
        "--stage",
        action="append",
        choices=["all", "prices", "regime", "fundamentals", "earnings"],
        default=None,
        help=(
            "Fetch only selected stage(s). May be passed multiple times. "
            "Default: all stages in prices -> regime -> fundamentals -> "
            "earnings order."
        ),
    )

    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip price download even when running all stages.",
    )

    parser.add_argument(
        "--skip-regime",
        action="store_true",
        help="Skip VIX/SPY regime download even when running all stages.",
    )

    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip fundamental snapshot download even when running all stages.",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing final parquet outputs and resume fundamentals "
            "from data/raw/fundamentals_checkpoint.parquet. "
            "(default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch selected stages even if their final parquet files exist.",
    )

    parser.add_argument(
        "--fundamentals-delay",
        type=float,
        default=5.0,
        help=(
            "Base seconds to wait between yfinance fundamental requests. "
            "Jitter is added automatically. (default: %(default)s)"
        ),
    )

    parser.add_argument(
        "--rate-limit-cooldown",
        type=int,
        default=900,
        help=(
            "Seconds to wait before retrying after a Yahoo rate-limit response. "
            "(default: %(default)s)"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(start_time: float) -> None:
    """
    Print a summary of what was written to data/raw/.

    Args:
        start_time: Unix timestamp from time.time() at script start.
    """
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  FETCH COMPLETE")
    print("=" * 60)
    print(f"  Elapsed time : {elapsed:.1f}s")
    print()

    files = {
        "prices.parquet": "Daily OHLCV for all candidate tickers",
        "fundamentals.parquet": "P/S ratio and revenue growth snapshot",
        "regime.parquet": "VIX and SPY daily close",
    }

    all_present = True
    for filename, description in files.items():
        path = RAW_DIR / filename
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK]  {filename:<30} {size_kb:>8.1f} KB  —  {description}")
        else:
            print(f"  [!!]  {filename:<30} MISSING        —  {description}")
            all_present = False

    if FETCH_STATUS_PATH.exists():
        print()
        print(f"  Fetch status: {FETCH_STATUS_PATH}")

    print()

    if all_present:
        print("  All files written successfully.")
        print()
        print("  Next step:")
        print("    uv run python scripts/02_build_universe.py")
    else:
        print("  WARNING: One or more files are missing.")
        print("  Check the logs above for errors and re-run.")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    setup_logging()
    args = parse_args()
    logger = logging.getLogger(__name__)

    print()
    print("=" * 60)
    print("  ARQ PAIRS TRADING — DATA FETCH")
    print("=" * 60)
    stages = _resolve_stages(args)

    print(f"  History       : {args.years} years")
    print(f"  Retries       : {args.retries} per ticker")
    print(f"  Stages        : {', '.join(stages)}")
    print(f"  Resume        : {'yes' if args.resume else 'no'}")
    print(f"  Force refetch : {'yes' if args.force else 'no'}")
    print(f"  Disable proxy : {'yes' if args.disable_proxy else 'no'}")
    print(f"  Output        : {RAW_DIR.resolve()}")
    print("=" * 60)
    print()

    start_time = time.time()

    try:
        if args.disable_proxy:
            os.environ["ARQ_YF_DISABLE_PROXY"] = "1"
        fetch_all(
            years=args.years,
            retry_attempts=args.retries,
            stages=set(stages),
            resume=args.resume,
            force=args.force,
            fundamentals_delay=args.fundamentals_delay,
            rate_limit_cooldown=args.rate_limit_cooldown,
        )
    except Exception as exc:
        logger.error("Fetch failed with an unexpected error: %s", exc)
        logger.error("Check your internet connection and try again.")
        sys.exit(1)

    print_summary(start_time)


def _resolve_stages(args: argparse.Namespace) -> list[str]:
    """
    Resolve requested fetch stages from --stage and --skip-* flags.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Ordered list of stages to run.
    """
    stage_order = ["prices", "regime", "fundamentals", "earnings"]
    requested = args.stage or ["all"]

    if "all" in requested:
        stages = stage_order.copy()
    else:
        stages = [stage for stage in stage_order if stage in requested]

    if args.skip_prices and "prices" in stages:
        stages.remove("prices")
    if args.skip_regime and "regime" in stages:
        stages.remove("regime")
    if args.skip_fundamentals and "fundamentals" in stages:
        stages.remove("fundamentals")

    if not stages:
        raise SystemExit("No fetch stages selected.")

    return stages


if __name__ == "__main__":
    main()