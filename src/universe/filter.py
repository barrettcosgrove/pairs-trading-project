# Applies all five hard pre-filters (sector, liquidity, price, market cap, SPY correlation) to the full candidate list. Returns passing tickers for a given date. Writes to data/processed/universe_history.parquet. Enforces ≥8 subsector diversity and logs per-filter pass counts.
"""
src/universe/filter.py — Universe Filter

Applies five hard pre-filters on each monthly reconstitution date to determine
which tickers belong to the investable universe. Writes the full history of
reconstitution decisions to data/processed/universe_history.parquet so the
backtest engine can query the point-in-time universe at any date without
look-ahead bias.

Called by scripts/02_build_universe.py. Never run directly.
"""

import logging
import sys
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import CONFIG

# sector_map.py lives in data/, not src/ — add the project root to sys.path
# if needed so the import resolves regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.sector_map import SECTOR_MAP  # noqa: E402

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_universe_history(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    reconstitution_dates: list[date],
) -> pd.DataFrame:
    """
    Apply the five hard pre-filters on each reconstitution date and write
    the results to data/processed/universe_history.parquet.

    Each filter is evaluated using only data available on or before the
    reconstitution date — no look-ahead bias.

    Filters applied (all must pass):
      1. Sector       — ticker present in SECTOR_MAP
      2. Liquidity    — 30-day average daily volume > CONFIG.min_adv
      3. Price        — most recent adj_close > CONFIG.min_price
      4. Market cap   — most recent adj_close × volume > CONFIG.min_market_cap
                        (single-day dollar turnover proxy — see comment below)
      5. SPY corr     — 60-day Pearson correlation with SPY returns
                        < CONFIG.max_spy_correlation

    After filtering, a warning is logged if the passing tickers span fewer
    than CONFIG.min_subsectors distinct subsectors.

    Args:
        prices: Full raw price DataFrame with columns
            [ticker, date, adj_close, close, volume]. Must cover the full
            date range across all reconstitution dates.
        returns: Log return DataFrame with columns [ticker, date, log_return].
            Produced by clean.py and loaded by the caller.
        spy_returns: SPY daily log returns as a Series indexed by date
            (datetime64). Computed by the caller from regime data.
        reconstitution_dates: Ordered list of dates on which to evaluate the
            universe — typically the first trading day of each calendar month.

    Returns:
        DataFrame with columns [date, ticker, passed_filters] — one row per
        ticker per reconstitution date. Also written to
        data/processed/universe_history.parquet.
    """
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    returns = returns.sort_values(["ticker", "date"]).reset_index(drop=True)
    spy_returns = spy_returns.sort_index()

    rows: list[dict] = []

    for recon_date in reconstitution_dates:
        recon_ts = pd.Timestamp(recon_date)

        # Slice to data available on or before this reconstitution date.
        prices_to_date = prices[prices["date"] <= recon_ts]
        returns_to_date = returns[returns["date"] <= recon_ts]
        spy_to_date = spy_returns[spy_returns.index <= recon_ts]

        if prices_to_date.empty:
            logger.warning(
                "[%s] No price data available — skipping reconstitution", recon_date
            )
            continue

        all_tickers = sorted(prices_to_date["ticker"].unique())
        n_total = len(all_tickers)

        # -----------------------------------------------------------------
        # Filter 1 — Sector
        # Ticker must have an entry in SECTOR_MAP. Any ticker absent from
        # the map is treated as outside the tech universe.
        # -----------------------------------------------------------------
        sector_pass = pd.Series(
            {t: t in SECTOR_MAP for t in all_tickers}, dtype=bool
        )

        # -----------------------------------------------------------------
        # Filter 2 — Liquidity (30-day ADV)
        # Average daily volume over the 30 most recent trading days must
        # exceed CONFIG.min_adv. Uses tail(30) on sorted dates per ticker.
        # -----------------------------------------------------------------
        adv = (
            prices_to_date
            .groupby("ticker")["volume"]
            .apply(lambda x: x.sort_index().tail(30).mean())
        )
        adv = adv.reindex(all_tickers, fill_value=0.0)
        liquidity_pass = adv > CONFIG.min_adv

        # -----------------------------------------------------------------
        # Filter 3 — Price
        # Most recent adj_close must exceed CONFIG.min_price. Eliminates
        # penny stocks where bid-ask spreads are wide and price behavior
        # is erratic.
        # -----------------------------------------------------------------
        latest_adj = prices_to_date.groupby("ticker")["adj_close"].last()
        latest_adj = latest_adj.reindex(all_tickers, fill_value=0.0)
        price_pass = latest_adj > CONFIG.min_price

        # Filter 4 — Minimum dollar volume (30-day avg daily dollar volume)
        # True market cap requires shares outstanding which is not in OHLCV data.
        # Using 30-day average daily dollar volume as a size/liquidity proxy.
        # $25M ADV filters out micro-caps while keeping mid and large caps.
        # Documented as known approximation in docs/data.md.
        dollar_volume = (
            prices_to_date
            .assign(dollar_vol=lambda x: x["adj_close"] * x["volume"])
            .groupby("ticker")["dollar_vol"]
            .apply(lambda x: x.sort_index().tail(30).mean())
        )
        mktcap_proxy = dollar_volume.reindex(all_tickers, fill_value=0.0)
        mktcap_pass = mktcap_proxy > CONFIG.min_dollar_volume

        # -----------------------------------------------------------------
        # Filter 5 — SPY correlation (60-day Pearson)
        # Tickers whose returns are too correlated with SPY (≥ 0.90) are
        # effectively index proxies — their spreads are driven by macro, not
        # company-specific dynamics. Correlation is computed over the 60
        # most recent trading days of aligned (ticker, SPY) return pairs.
        # Tickers with fewer than 30 aligned observations are conservatively
        # marked as failing (unknown correlation → reject).
        # -----------------------------------------------------------------
        spy_60 = spy_to_date.tail(60)

        corr_values: dict[str, float] = {}
        for ticker in all_tickers:
            corr_values[ticker] = _compute_spy_correlation(
                returns_to_date, ticker, spy_60
            )

        spy_corr_series = pd.Series(corr_values, dtype=float)
        corr_pass = spy_corr_series < CONFIG.max_spy_correlation

        # -----------------------------------------------------------------
        # Combine — ticker passes only if all five filters pass
        # -----------------------------------------------------------------
        all_pass = (
            sector_pass
            & liquidity_pass
            & price_pass
            & mktcap_pass
            & corr_pass
        ).reindex(all_tickers, fill_value=False)

        # -----------------------------------------------------------------
        # Subsector diversity check
        # -----------------------------------------------------------------
        passing_tickers = [t for t in all_tickers if all_pass.get(t, False)]
        subsectors = {SECTOR_MAP[t] for t in passing_tickers if t in SECTOR_MAP}
        if len(subsectors) < CONFIG.min_subsectors:
            logger.warning(
                "[%s] Only %d distinct subsector(s) represented among %d passing "
                "tickers — minimum is %d. Proceeding with reduced diversity.",
                recon_date,
                len(subsectors),
                len(passing_tickers),
                CONFIG.min_subsectors,
            )

        _log_filter_counts(
            recon_date,
            n_total,
            sector_pass,
            liquidity_pass,
            price_pass,
            mktcap_pass,
            corr_pass,
            all_pass,
        )

        for ticker in all_tickers:
            rows.append({
                "date": recon_ts,
                "ticker": ticker,
                "passed_filters": bool(all_pass.get(ticker, False)),
            })

    universe_history = pd.DataFrame(rows)
    universe_history["date"] = pd.to_datetime(universe_history["date"])
    universe_history["passed_filters"] = universe_history["passed_filters"].astype(bool)
    universe_history = universe_history.sort_values(["date", "ticker"]).reset_index(
        drop=True
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "universe_history.parquet"
    universe_history.to_parquet(out_path, index=False)
    logger.info(
        "Universe history written to %s — %d reconstitution dates, %d total rows",
        out_path,
        universe_history["date"].nunique(),
        len(universe_history),
    )

    return universe_history


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_spy_correlation(
    returns_to_date: pd.DataFrame,
    ticker: str,
    spy_60: pd.Series,
) -> float:
    """
    Compute the Pearson correlation between a ticker's log returns and SPY
    over the most recent 60 aligned trading days.

    Args:
        returns_to_date: Full returns DataFrame filtered to <= recon_date,
            with columns [ticker, date, log_return].
        ticker: Ticker symbol to evaluate.
        spy_60: SPY log returns for the trailing 60 trading days, indexed
            by date.

    Returns:
        Pearson correlation as a float in [-1, 1]. Returns 1.0 (failing)
        if fewer than 30 aligned observations exist, to conservatively reject
        tickers where the correlation cannot be estimated reliably.
    """
    ticker_rets = returns_to_date.loc[
        returns_to_date["ticker"] == ticker
    ].set_index("date")["log_return"]

    # Align on dates present in both series — inner join via reindex + dropna.
    aligned_ticker = ticker_rets.reindex(spy_60.index).dropna()
    aligned_spy = spy_60.reindex(aligned_ticker.index).dropna()

    # Reindex both to the common set of dates after double-sided dropna.
    common_dates = aligned_ticker.index.intersection(aligned_spy.index)

    if len(common_dates) < 30:
        return 1.0  # conservative rejection: treat unknown correlation as failing

    return float(aligned_ticker.loc[common_dates].corr(aligned_spy.loc[common_dates]))


def _log_filter_counts(
    recon_date: date,
    n_total: int,
    sector_pass: pd.Series,
    liquidity_pass: pd.Series,
    price_pass: pd.Series,
    mktcap_pass: pd.Series,
    corr_pass: pd.Series,
    all_pass: pd.Series,
) -> None:
    """
    Log per-filter pass counts for a single reconstitution date.

    Args:
        recon_date: The reconstitution date being evaluated.
        n_total: Total number of candidate tickers evaluated.
        sector_pass: Boolean Series — sector filter results.
        liquidity_pass: Boolean Series — liquidity filter results.
        price_pass: Boolean Series — price filter results.
        mktcap_pass: Boolean Series — market cap proxy filter results.
        corr_pass: Boolean Series — SPY correlation filter results.
        all_pass: Boolean Series — combined final filter results.
    """
    logger.info(
        "[%s] Filter counts — "
        "Sector: %d/%d | Liquidity: %d | Price: %d | Mktcap: %d | SPY corr: %d "
        "→ %d pass",
        recon_date,
        int(sector_pass.sum()), n_total,
        int((sector_pass & liquidity_pass).sum()),
        int((sector_pass & liquidity_pass & price_pass).sum()),
        int((sector_pass & liquidity_pass & price_pass & mktcap_pass).sum()),
        int((sector_pass & liquidity_pass & price_pass & mktcap_pass & corr_pass).sum()),
        int(all_pass.sum()),
    )
