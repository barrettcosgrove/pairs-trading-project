# Calls yfinance to download prices, fundamentals, VIX, and SPY. Writes results to data/raw/. Includes retry logic and logs fetched tickers and failures.
"""
src/data/fetch.py — Data Fetcher

Downloads all raw data from yfinance and writes it to data/raw/ as parquet
files. Designed to run once at project start via scripts/01_fetch_data.py,
and re-run only for quarterly refreshes.

Nothing outside this file should import yfinance. All other modules read
data through src/data/load.py.

Outputs:
    data/raw/prices.parquet      — Daily OHLCV for all candidate tickers
    data/raw/fundamentals.parquet — P/S ratio and revenue growth snapshot
    data/raw/regime.parquet      — Daily VIX close and SPY adjusted close

Known limitations:
    - yfinance only returns currently-listed stocks (survivorship bias)
    - Fundamental data is a current snapshot, not point-in-time historical
    - Split/dividend adjustment methodology is yfinance default
    See docs/data.md for full discussion.
"""

import io
import json
import logging
import os
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as cffi_requests
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)

_yf_session: cffi_requests.Session | None = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
FETCH_STATUS_PATH = RAW_DIR / "fetch_status.json"
FUNDAMENTALS_CHECKPOINT_PATH = RAW_DIR / "fundamentals_checkpoint.parquet"
FETCH_MANIFEST_PATH = RAW_DIR / "fetch_manifest.json"


# ---------------------------------------------------------------------------
# Candidate Universe
# ---------------------------------------------------------------------------

# Full list of ~95 candidate tech tickers to fetch.
# Superset of the final universe — hard filters in src/universe/filter.py
# will trim this down to the target 100.
# Includes some historically delisted/acquired names where data still exists
# to partially mitigate survivorship bias.


CANDIDATE_TICKERS: list[str] = [
    # Semiconductors (8)
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT",
    
    # Cloud / Enterprise Software (10 - S&P 500 only)
    "MSFT", "CRM", "NOW", "ADSK", "CDNS", "SNPS", "WDAY", "ORCL", "ADBE", "ANET",
    
    # Cybersecurity (4 - the ones in S&P 500)
    "FTNT", "PANW", "CRWD", "CHKP",
    
    # Energy (8)
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
    
    # Financials (9)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW",
    
    # Healthcare (10)
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BSX",
    
    # Consumer Staples (8)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "CL", "GIS",
    
    # Industrials (10)
    "CAT", "DE", "HON", "GE", "LMT", "UPS", "FDX", "EMR", "ETN", "RTX",
    
    # Utilities (8)
    "NEE", "DUK", "SO", "AEP", "EXC", "D", "XEL", "WEC",
    
    # Materials (6)
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX",
    
    # Consumer Discretionary (8)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    
    # Communication Services (6)    
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T",
]

print(f"Total S&P 500 tickers: {len(CANDIDATE_TICKERS)}")  # 95

# Deduplicate while preserving order
CANDIDATE_TICKERS = list(dict.fromkeys(CANDIDATE_TICKERS))

# Regime and benchmark tickers — always fetched regardless of universe filters
REGIME_TICKERS: list[str] = ["^VIX", "SPY"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_all(
    years: float = 3.5,
    retry_attempts: int = 3,
    stages: set[str] | None = None,
    resume: bool = True,
    force: bool = False,
    fundamentals_delay: float = 5.0,
    rate_limit_cooldown: int = 900,
) -> None:
    """
    Download all raw data and write to data/raw/.

    Fetches prices and fundamentals for all candidate tickers, plus VIX
    and SPY for regime filtering. Skips tickers that fail after retries
    and logs them for review.

    Args:
        years: Number of years of history to fetch. Default 3.5 gives
               2.5 years for training + 6 months OOS + buffer for
               120-day clustering window warmup.
        retry_attempts: Number of times to retry a failed ticker fetch.
        stages: Stages to run. Valid values are "prices", "regime", and
            "fundamentals". Defaults to all stages.
        resume: Reuse completed outputs/checkpoints where possible.
        force: Refetch stages even if their final parquet already exists.
        fundamentals_delay: Base delay between fundamental ticker requests.
        rate_limit_cooldown: Seconds to wait before retrying after a Yahoo
            rate-limit response.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _configure_yfinance_runtime()

    end_date = date(2026, 4, 1)
    start_date = date(2019, 1, 1)

    logger.info(
        "Fetching data from %s to %s for %d candidate tickers",
        start_date,
        end_date,
        len(CANDIDATE_TICKERS),
    )

    selected_stages = stages or {"prices", "regime", "fundamentals"}
    regime_sources: dict[str, str] = {"vix": "unknown", "spy": "unknown"}

    if "prices" in selected_stages:
        if _should_skip_stage(RAW_DIR / "prices.parquet", resume, force):
            logger.info("Skipping prices; data/raw/prices.parquet already exists.")
            _write_fetch_status("prices", "skipped", message="output already exists")
        else:
            fetch_prices(CANDIDATE_TICKERS, start_date, end_date, retry_attempts)

    # Regime data is small and required by universe filtering, so fetch it
    # before fundamentals, which are the most rate-limit-prone Yahoo endpoint.
    if "regime" in selected_stages:
        if _should_skip_stage(RAW_DIR / "regime.parquet", resume, force):
            logger.info("Skipping regime; data/raw/regime.parquet already exists.")
            _write_fetch_status("regime", "skipped", message="output already exists")
        else:
            regime_sources = fetch_regime(start_date, end_date, retry_attempts)

    if "fundamentals" in selected_stages:
        if _should_skip_stage(RAW_DIR / "fundamentals.parquet", resume, force):
            logger.info(
                "Skipping fundamentals; data/raw/fundamentals.parquet already exists."
            )
            _write_fetch_status(
                "fundamentals", "skipped", message="output already exists"
            )
        else:
            fetch_fundamentals(
                CANDIDATE_TICKERS,
                retry_attempts,
                resume=resume,
                request_delay=fundamentals_delay,
                rate_limit_cooldown=rate_limit_cooldown,
            )

    logger.info("All fetches complete. Files written to %s", RAW_DIR)
    _write_fetch_manifest(regime_sources["vix"], regime_sources["spy"])


def fetch_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    retry_attempts: int = 3,
) -> None:
    """
    Download daily OHLCV for all candidate tickers in a single batch
    request and write to data/raw/prices.parquet.

    Uses yfinance batch download instead of per-ticker requests to avoid
    Yahoo Finance rate limiting. Yahoo treats the full ticker list as one
    request, returns a MultiIndex DataFrame keyed by ticker, which we
    reshape into the same flat long format as before.

    Args:
        tickers: List of ticker symbols to fetch.
        start_date: First date of history to fetch (inclusive).
        end_date: Last date of history to fetch (inclusive).
        retry_attempts: Number of retries on failure.
    """
    logger.info(
        "Batch downloading %d tickers from %s to %s",
        len(tickers), start_date, end_date,
    )

    # Single batch call — replaces the old per-ticker loop that triggered
    # rate limiting after the first few requests.
    for attempt in range(1, retry_attempts + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=False,
                progress=True,      # shows a progress bar in terminal
                group_by="ticker",  # organises MultiIndex columns by ticker
                threads=True,       # yfinance handles parallelism safely
                session=_get_yf_session(),
            )

            if raw.empty:
                raise ValueError("Batch download returned empty DataFrame")

            break  # success — exit retry loop

        except Exception as exc:
            logger.warning(
                "Batch download attempt %d/%d failed: %s",
                attempt, retry_attempts, exc,
            )
            if attempt < retry_attempts:
                # Linear backoff: 10s, 20s, 30s — batch failures are usually
                # transient network issues, not per-ticker problems.
                wait = 10 * attempt
                logger.info("Waiting %ds before retry...", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    "Batch price download failed after "
                    f"{retry_attempts} attempts"
                ) from exc

    # Reshape MultiIndex DataFrame into flat long format.
    # raw has columns like (AAPL, Close), (AAPL, Open), (NVDA, Close)...
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                # Single-ticker download returns flat columns, not MultiIndex.
                ticker_df = raw.copy()
            else:
                ticker_df = raw[ticker].copy()

            # Rows where Adj Close is NaN mean the ticker had no data that day.
            ticker_df = ticker_df.dropna(subset=["Adj Close"])

            if ticker_df.empty:
                logger.warning("No data returned for %s — skipping", ticker)
                failed.append(ticker)
                continue

            df = pd.DataFrame({
                "ticker":    ticker,
                "date":      pd.to_datetime(ticker_df.index),
                "open":      ticker_df["Open"].values,
                "high":      ticker_df["High"].values,
                "low":       ticker_df["Low"].values,
                "close":     ticker_df["Close"].values,
                "adj_close": ticker_df["Adj Close"].values,
                "volume":    ticker_df["Volume"].values,
            })

            frames.append(df)

        except KeyError:
            logger.warning("Ticker %s missing from batch response — skipping", ticker)
            failed.append(ticker)

    if not frames:
        raise RuntimeError("No price data was successfully parsed from batch download.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    out_path = RAW_DIR / "prices.parquet"
    prices.to_parquet(out_path, index=False)

    logger.info(
        "Prices written to %s — %d tickers, %d rows",
        out_path,
        prices["ticker"].nunique(),
        len(prices),
    )

    if failed:
        logger.warning(
            "%d tickers had no data and were skipped: %s",
            len(failed), ", ".join(failed),
        )

    _log_summary("prices", prices, failed)


def fetch_fundamentals(
    tickers: list[str],
    retry_attempts: int = 3,
    resume: bool = True,
    request_delay: float = 5.0,
    rate_limit_cooldown: int = 900,
) -> None:
    """
    Download current P/S ratio and TTM revenue growth for all candidate
    tickers and write to data/raw/fundamentals.parquet.

    Note: This is a current snapshot, not point-in-time historical data.
    Fundamental scores are held constant over the backtest as a known
    simplification. See docs/data.md for details.

    Args:
        tickers: List of ticker symbols to fetch.
        retry_attempts: Number of retries on failure per ticker.
        resume: Continue from data/raw/fundamentals_checkpoint.parquet if it
            exists.
        request_delay: Base delay between ticker requests.
        rate_limit_cooldown: Seconds to wait before retrying after a Yahoo
            rate-limit response.
    """
    records = _load_fundamentals_checkpoint() if resume else []
    completed = {record["ticker"] for record in records}
    failed: list[str] = []
    fetch_date = date.today()
    last_ticker: str | None = None

    if completed:
        logger.info(
            "Resuming fundamentals from checkpoint with %d completed tickers",
            len(completed),
        )

    for i, ticker in enumerate(tickers):
        if ticker in completed:
            logger.info(
                "Skipping fundamentals for %s (%d/%d); already checkpointed",
                ticker, i + 1, len(tickers),
            )
            continue

        last_ticker = ticker
        pending = [candidate for candidate in tickers[i:] if candidate not in completed]
        _write_fetch_status(
            "fundamentals",
            "running",
            completed_tickers=sorted(completed),
            failed_tickers=failed,
            last_ticker=last_ticker,
            next_ticker=ticker,
            pending_tickers=pending,
        )

        logger.info(
            "Fetching fundamentals: %s (%d/%d)", ticker, i + 1, len(tickers)
        )

        try:
            record = _fetch_ticker_fundamentals(
                ticker,
                fetch_date,
                retry_attempts,
                rate_limit_cooldown,
            )
        except YFRateLimitError as exc:
            _write_fundamentals_checkpoint(records)
            _write_fetch_status(
                "fundamentals",
                "rate_limited",
                completed_tickers=sorted(completed),
                failed_tickers=failed,
                last_ticker=last_ticker,
                next_ticker=ticker,
                pending_tickers=[candidate for candidate in tickers[i:] if candidate not in completed],
                message=(
                    "Yahoo rate limited fundamentals. Wait before resuming with "
                    "`uv run python scripts/01_fetch_data.py --stage fundamentals --resume`."
                ),
            )
            raise RuntimeError(
                "Yahoo rate-limited fundamentals at "
                f"{ticker}. Check {FETCH_STATUS_PATH} for resume details."
            ) from exc

        if record is not None:
            records.append(record)
            completed.add(ticker)
            _write_fundamentals_checkpoint(records)
        else:
            failed.append(ticker)
            logger.warning(
                "Failed to fetch fundamentals for %s — skipping", ticker
            )

        # Ticker.info uses Yahoo's quote-summary endpoint, which is much more
        # rate-limited than price downloads. Jitter avoids a fixed scrape cadence.
        time.sleep(request_delay + random.uniform(0, request_delay * 0.4))

    if not records:
        raise RuntimeError("No fundamental data was successfully fetched.")

    fundamentals = pd.DataFrame(records)
    out_path = RAW_DIR / "fundamentals.parquet"
    fundamentals.to_parquet(out_path, index=False)

    logger.info(
        "Fundamentals written to %s — %d tickers",
        out_path,
        len(fundamentals),
    )
    _write_fetch_status(
        "fundamentals",
        "completed",
        completed_tickers=sorted(completed),
        failed_tickers=failed,
        last_ticker=last_ticker,
        next_ticker=None,
        pending_tickers=[],
    )

    if failed:
        logger.warning(
            "%d tickers missing fundamental data: %s",
            len(failed),
            ", ".join(failed),
        )


def fetch_vix_cboe(start_date: date) -> pd.DataFrame:
    """
    Download daily VIX history from the CBOE CDN and return a filtered DataFrame.

    Args:
        start_date: Earliest date to include, inclusive.

    Returns:
        DataFrame with columns [date, vix], date as datetime64[ns],
        rows filtered to >= start_date. Source: CBOE (no API key required).
    """
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    logger.info("Fetching VIX history from CBOE: %s", url)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"CBOE VIX download returned HTTP {exc.response.status_code}: {url}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"CBOE VIX download failed: {exc}") from exc

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CBOE VIX CSV: {exc}") from exc

    df.columns = [c.strip().lower() for c in df.columns]
    if "close" not in df.columns or "date" not in df.columns:
        raise RuntimeError(
            f"Unexpected CBOE VIX CSV columns: {list(df.columns)}"
        )

    df = df.rename(columns={"close": "vix"})[["date", "vix"]].dropna()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start_date)]
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "VIX data from CBOE — %d rows from %s",
        len(df),
        df["date"].min().date() if not df.empty else "N/A",
    )
    return df[["date", "vix"]]


def fetch_spy_alphavantage(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download daily SPY adjusted close from Alpha Vantage and return a filtered DataFrame.

    Args:
        start_date: Earliest date to include, inclusive.
        end_date: Latest date to include, inclusive.

    Returns:
        DataFrame with columns [date, spy], date as datetime64[ns],
        rows filtered to [start_date, end_date].
    """
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        raise RuntimeError("ALPHA_VANTAGE_KEY environment variable is not set")

    logger.info(
        "Fetching SPY data from Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED, outputsize=full)"
    )

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": "SPY",
        "outputsize": "full",
        "apikey": api_key,
        "datatype": "json",
    }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query", params=params, timeout=60
        )
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Alpha Vantage returned HTTP {exc.response.status_code}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Alpha Vantage SPY download failed: {exc}") from exc

    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    if "Information" in data:
        raise RuntimeError(
            f"Alpha Vantage rate limit or info message: {data['Information']}"
        )
    if "Time Series (Daily)" not in data:
        raise RuntimeError(
            f"Alpha Vantage response missing 'Time Series (Daily)' key; "
            f"got keys: {list(data.keys())}"
        )

    ts = data["Time Series (Daily)"]
    try:
        records = [
            {"date": pd.to_datetime(d), "spy": float(v["5. adjusted close"])}
            for d, v in ts.items()
        ]
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse Alpha Vantage time series: {exc}"
        ) from exc

    df = pd.DataFrame(records)
    df = df[
        (df["date"] >= pd.Timestamp(start_date))
        & (df["date"] <= pd.Timestamp(end_date))
    ]
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("SPY data from Alpha Vantage — %d rows", len(df))
    return df[["date", "spy"]]


def fetch_regime(
    start_date: date,
    end_date: date,
    retry_attempts: int = 3,
) -> dict[str, str]:
    """
    Download daily VIX close and SPY adjusted close and write to
    data/raw/regime.parquet.

    Tries CBOE for VIX (no key required) and falls back to yfinance.
    Tries Alpha Vantage for SPY if ALPHA_VANTAGE_KEY is set, falls back
    to yfinance, then synthetic data as a last resort.

    Used by:
        - src/regime/vix.py for the portfolio-level VIX filter
        - src/universe/filter.py for the SPY correlation pre-filter

    Args:
        start_date: First date of history to fetch (inclusive).
        end_date: Last date of history to fetch (inclusive).
        retry_attempts: Number of retries on failure.

    Returns:
        Dict mapping 'vix' and 'spy' to their source labels,
        e.g. {'vix': 'cboe', 'spy': 'alphavantage'}.
    """
    # --- VIX: try CBOE first, fall back to yfinance ---
    vix_source = "cboe"
    try:
        vix_df = fetch_vix_cboe(start_date)
        logger.info("VIX source: %s (%d rows)", vix_source, len(vix_df))
    except RuntimeError as exc:
        logger.warning("CBOE VIX failed (%s) — falling back to yfinance", exc)
        vix_source = "yfinance"
        vix_df = _fetch_yfinance_regime_ticker(
            "^VIX", "vix", start_date, end_date, retry_attempts
        )
        logger.info("VIX source: %s (%d rows)", vix_source, len(vix_df))

    # --- SPY: Alpha Vantage → yfinance → synthetic ---
    if os.getenv("ALPHA_VANTAGE_KEY"):
        spy_source = "alphavantage"
        try:
            spy_df = fetch_spy_alphavantage(start_date, end_date)
            logger.info("SPY source: %s (%d rows)", spy_source, len(spy_df))
        except RuntimeError as exc:
            logger.warning(
                "Alpha Vantage SPY failed (%s) — falling back to yfinance", exc
            )
            spy_source = "yfinance"
            try:
                spy_df = _fetch_yfinance_regime_ticker(
                    "SPY", "spy", start_date, end_date, retry_attempts
                )
                logger.info("SPY source: %s (%d rows)", spy_source, len(spy_df))
            except RuntimeError as exc2:
                logger.warning(
                    "yfinance SPY also failed (%s) — using synthetic data", exc2
                )
                spy_source = "synthetic"
                spy_df = _make_synthetic_spy(start_date, end_date)
                logger.info("SPY source: %s (%d rows)", spy_source, len(spy_df))
    else:
        logger.info("ALPHA_VANTAGE_KEY not set; fetching SPY from yfinance")
        spy_source = "yfinance"
        try:
            spy_df = _fetch_yfinance_regime_ticker(
                "SPY", "spy", start_date, end_date, retry_attempts
            )
            logger.info("SPY source: %s (%d rows)", spy_source, len(spy_df))
        except RuntimeError as exc:
            logger.warning(
                "yfinance SPY failed (%s) — using synthetic data", exc
            )
            spy_source = "synthetic"
            spy_df = _make_synthetic_spy(start_date, end_date)
            logger.info("SPY source: %s (%d rows)", spy_source, len(spy_df))

    regime = pd.merge(vix_df, spy_df, on="date", how="inner")
    regime = regime.sort_values("date").reset_index(drop=True)

    out_path = RAW_DIR / "regime.parquet"
    regime.to_parquet(out_path, index=False)

    logger.info(
        "Regime data written to %s — %d rows, vix_source=%s, spy_source=%s",
        out_path,
        len(regime),
        vix_source,
        spy_source,
    )
    _write_fetch_status("regime", "completed")
    return {"vix": vix_source, "spy": spy_source}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_ticker_fundamentals(
    ticker: str,
    fetch_date: date,
    retry_attempts: int,
    rate_limit_cooldown: int,
) -> dict | None:
    """
    Fetch current P/S ratio and TTM revenue growth for a single ticker.

    Args:
        ticker: Ticker symbol.
        fetch_date: Date of the snapshot (today).
        retry_attempts: Max attempts before returning None.
        rate_limit_cooldown: Seconds to wait before retrying after a Yahoo
            rate-limit response.

    Returns:
        Dict with keys [ticker, fetch_date, price_to_sales,
        revenue_growth_ttm], or None if all attempts fail or data
        is unavailable.
    """
    for attempt in range(1, retry_attempts + 1):
        try:
            info = yf.Ticker(ticker, session=_get_yf_session()).info

            # yfinance returns None for missing fields — treat as unavailable
            price_to_sales = info.get("priceToSalesTrailing12Months")
            revenue_growth = info.get("revenueGrowth")

            if price_to_sales is None and revenue_growth is None:
                logger.warning(
                    "No fundamental data available for %s", ticker
                )
                return None

            return {
                "ticker": ticker,
                "fetch_date": fetch_date,
                "price_to_sales": price_to_sales,
                "revenue_growth_ttm": revenue_growth,
            }

        except YFRateLimitError:
            logger.warning(
                "Attempt %d/%d rate-limited for %s fundamentals",
                attempt,
                retry_attempts,
                ticker,
            )
            if attempt < retry_attempts:
                logger.info("Waiting %ds before retrying fundamentals...", rate_limit_cooldown)
                time.sleep(rate_limit_cooldown)
            else:
                raise

        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for %s fundamentals: %s",
                attempt,
                retry_attempts,
                ticker,
                exc,
            )
            if attempt < retry_attempts:
                time.sleep(2 ** attempt)

    return None


def _should_skip_stage(path: Path, resume: bool, force: bool) -> bool:
    """
    Decide whether a fetch stage can be skipped.

    Args:
        path: Final parquet output for the stage.
        resume: Whether existing outputs should be reused.
        force: Whether the stage should be refetched.

    Returns:
        True if the stage should be skipped.
    """
    return resume and not force and path.exists()


def _load_fundamentals_checkpoint() -> list[dict]:
    """
    Load partial fundamentals from the checkpoint parquet.

    Returns:
        List of checkpoint records, or an empty list if no checkpoint exists.
    """
    if not FUNDAMENTALS_CHECKPOINT_PATH.exists():
        return []

    checkpoint = pd.read_parquet(FUNDAMENTALS_CHECKPOINT_PATH)
    return checkpoint.to_dict(orient="records")


def _write_fundamentals_checkpoint(records: list[dict]) -> None:
    """
    Persist partial fundamentals so a rate-limited run can resume later.

    Args:
        records: Fundamental records collected so far.
    """
    if not records:
        return

    checkpoint = pd.DataFrame(records).drop_duplicates(
        subset=["ticker"], keep="last"
    )
    checkpoint.to_parquet(FUNDAMENTALS_CHECKPOINT_PATH, index=False)


def _write_fetch_status(
    stage: str,
    status: str,
    *,
    completed_tickers: list[str] | None = None,
    failed_tickers: list[str] | None = None,
    last_ticker: str | None = None,
    next_ticker: str | None = None,
    pending_tickers: list[str] | None = None,
    message: str | None = None,
) -> None:
    """
    Write human-readable machine-parsable fetch progress metadata.

    Args:
        stage: Fetch stage being updated.
        status: Stage status, e.g. "running", "completed", "rate_limited".
        completed_tickers: Tickers successfully fetched so far.
        failed_tickers: Tickers skipped due to missing data or non-rate-limit
            failures.
        last_ticker: Most recent ticker attempted.
        next_ticker: Ticker to try first on resume.
        pending_tickers: Tickers not yet completed.
        message: Optional status message for humans.
    """
    payload = {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "stage": stage,
        "status": status,
        "completed_count": len(completed_tickers or []),
        "failed_count": len(failed_tickers or []),
        "pending_count": len(pending_tickers or []),
        "last_ticker": last_ticker,
        "next_ticker": next_ticker,
        "completed_tickers": completed_tickers or [],
        "failed_tickers": failed_tickers or [],
        "pending_tickers": pending_tickers or [],
        "message": message,
        "checkpoint_path": str(FUNDAMENTALS_CHECKPOINT_PATH)
        if stage == "fundamentals"
        else None,
    }

    FETCH_STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def _log_summary(
    data_type: str,
    df: pd.DataFrame,
    failed: list[str],
) -> None:
    """
    Log a fetch summary with date range and row counts.

    Args:
        data_type: Label for the log message (e.g. "prices").
        df: The fetched DataFrame.
        failed: List of tickers that failed to fetch.
    """
    if "date" in df.columns:
        logger.info(
            "%s summary — date range: %s to %s | tickers: %d | failed: %d",
            data_type,
            df["date"].min().date(),
            df["date"].max().date(),
            df["ticker"].nunique() if "ticker" in df.columns else "N/A",
            len(failed),
        )


def _configure_yfinance_runtime() -> None:
    """
    Configure cache and proxy session for yfinance.

    Environment controls:
        ARQ_YF_DISABLE_PROXY: If "1", create a requests.Session with
            trust_env=False to bypass both env-var and OS-level proxy
            settings (e.g. macOS System Settings → Network → Proxies).
        ARQ_YF_PROXY: If set, create a session that routes through this
            explicit proxy URL.
    """
    global _yf_session

    cache_dir = RAW_DIR / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    disable_proxy = os.getenv("ARQ_YF_DISABLE_PROXY", "0") == "1"
    explicit_proxy = os.getenv("ARQ_YF_PROXY", "").strip() or None

    if disable_proxy:
        session = cffi_requests.Session(impersonate="chrome")
        session.trust_env = False  # Ignore env vars AND OS system proxy
        _yf_session = session
        logger.info("Proxy bypass active: using trust_env=False curl_cffi session.")
    elif explicit_proxy:
        session = cffi_requests.Session()
        session.proxies.update({"http": explicit_proxy, "https": explicit_proxy})
        _yf_session = session
        logger.info("Using explicit yfinance proxy from ARQ_YF_PROXY.")
    else:
        _yf_session = None  # yfinance default behaviour


def _get_yf_session() -> cffi_requests.Session | None:
    """
    Return the configured yfinance session, or None for default behaviour.

    Returns:
        Session with trust_env=False if --disable-proxy was used, a session
        with explicit proxy if ARQ_YF_PROXY is set, or None otherwise.
    """
    return _yf_session


def _fetch_yfinance_regime_ticker(
    ticker: str,
    col_name: str,
    start_date: date,
    end_date: date,
    retry_attempts: int,
) -> pd.DataFrame:
    """
    Download a single regime ticker via yfinance with retry logic.

    Args:
        ticker: yfinance symbol, e.g. "^VIX" or "SPY".
        col_name: Column name for the close values in the returned DataFrame.
        start_date: First date of history to fetch (inclusive).
        end_date: Last date of history to fetch (inclusive).
        retry_attempts: Number of retries on failure.

    Returns:
        DataFrame with columns [date, col_name], date as datetime64[ns].
    """
    for attempt in range(1, retry_attempts + 1):
        try:
            raw = yf.download(
                tickers=[ticker],
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
                session=_get_yf_session(),
            )

            if raw.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

            # Single-ticker list with group_by="ticker" returns a MultiIndex.
            try:
                ticker_df = raw[ticker].copy()
            except KeyError:
                ticker_df = raw.copy()

            close = ticker_df["Close"].dropna().squeeze()
            if close.empty:
                raise ValueError(f"yfinance returned no Close data for {ticker}")

            df = pd.DataFrame({
                "date": pd.to_datetime(close.index),
                col_name: close.values,
            })
            return df

        except Exception as exc:
            is_rate_limit = (
                "rate" in str(exc).lower()
                or "429" in str(exc)
                or "too many" in str(exc).lower()
            )
            wait = 60 if is_rate_limit else 2 ** attempt
            logger.warning(
                "yfinance regime ticker %s attempt %d/%d %s — waiting %ds: %s",
                ticker,
                attempt,
                retry_attempts,
                "rate-limited" if is_rate_limit else "failed",
                wait,
                exc,
            )
            if attempt < retry_attempts:
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"yfinance fallback for {ticker} failed after {retry_attempts} attempts"
                ) from exc

    raise RuntimeError(f"yfinance fallback for {ticker} failed after {retry_attempts} attempts")


def _make_synthetic_spy(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Generate synthetic SPY prices as a seeded random walk on business days.

    Args:
        start_date: First date of the price series.
        end_date: Last date of the price series.

    Returns:
        DataFrame with columns [date, spy], date as datetime64[ns].
    """
    logger.warning(
        "SYNTHETIC SPY DATA in use — no real SPY source was available. "
        "This is a random walk seeded at 42 and is only suitable for pipeline testing."
    )
    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) == 0:
        raise RuntimeError(
            f"No business days in range {start_date} to {end_date} for synthetic SPY"
        )
    rng = np.random.default_rng(42)
    prices = 400.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(dates)))
    return pd.DataFrame({"date": pd.to_datetime(dates), "spy": prices})


def _write_fetch_manifest(vix_source: str, spy_source: str) -> None:
    """
    Write a JSON manifest summarising data sources and row counts for all fetched files.

    Args:
        vix_source: Label for the VIX data source, e.g. "cboe" or "yfinance".
        spy_source: Label for the SPY data source, e.g. "alphavantage", "yfinance",
            or "synthetic".
    """
    def _row_count(path: Path) -> int | None:
        if not path.exists():
            return None
        return len(pd.read_parquet(path))

    payload = {
        "fetched_at": pd.Timestamp.utcnow().isoformat(),
        "prices": {
            "source": "yfinance",
            "row_count": _row_count(RAW_DIR / "prices.parquet"),
        },
        "regime": {
            "vix_source": vix_source,
            "spy_source": spy_source,
            "row_count": _row_count(RAW_DIR / "regime.parquet"),
        },
        "fundamentals": {
            "source": "yfinance",
            "row_count": _row_count(RAW_DIR / "fundamentals.parquet"),
        },
    }

    FETCH_MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Fetch manifest written to %s", FETCH_MANIFEST_PATH)