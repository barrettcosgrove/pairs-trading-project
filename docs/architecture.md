# System Architecture

This document describes the **implemented** pipeline: modules, data flow,
and contracts. Live parameters are in `src/config.py`. The original product
spec is [`strategy.md`](strategy.md) (several sections there — percentiles,
tiering, Bollinger — are not what the engine does). Empirical issues:
[`diagnostics.md`](diagnostics.md). Install and CLI: [`README.md`](../README.md).
Directory tree: [`file-structure.md`](file-structure.md). Parquet schemas:
[`data.md`](data.md).

If this file and the code disagree, trust the code and update this file.

---

## 1. Design Philosophy

**Pipeline over monolith.** Raw data → clean → universe → clusters →
scored pairs → signals → trades. Each step is a module with a typed
interface.

**Disk caches at fetch and universe only.** Prices, regime, fundamentals,
returns, and universe history live on disk. Clustering and scoring run
inside the backtest loop (quarterly). Correlation matrices may be cached
under `data/processed/correlation_matrices/` when `correlation.py` writes
them.

**Centralize configuration.** Every tunable is on the frozen
`StrategyConfig` in `src/config.py`.

---

## 2. Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL SOURCES                                           │
│  yfinance (prices, fundamentals, earnings)                  │
│  CBOE / yfinance (VIX); Alpha Vantage / yfinance (SPY)      │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER          src/data/                              │
│  fetch.py     — only file that imports yfinance / curl_cffi │
│  clean.py     — quality gates, writes returns.parquet       │
│  load.py      — only production parquet reader              │
└────────────────────────────┬────────────────────────────────┘
                             │  start/end may be None (full history)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIVERSE            src/universe/filter.py                 │
│  Monthly: sector_map + ADV + price + $ volume + SPY ρ       │
│  Writes data/processed/universe_history.parquet             │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  CLUSTERING          src/clustering/                        │
│  correlation.py — D = 1 − ρ over clustering_window          │
│  kmeans.py      — silhouette k ∈ [k_min, k_max]             │
│  Cadence: CONFIG.clustering_refresh_days                    │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SCORING             src/scoring/                           │
│  corr / coint / half-life / vol / sector weights            │
│  Hard gate: half-life in [halflife_min, halflife_max]       │
│  Soft floor: min_cointegration_score; β_F > min_formation_beta │
│  Up to finalists_per_cluster pairs if composite ≥ floor     │
└────────────────────────────┬────────────────────────────────┘
                             │  ticker_a, ticker_b, cluster_id,
                             │  composite_score, beta/mean/std_formation,
                             │  halflife_value
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  SIGNALS  src/signals/   │   │  REGIME  src/regime/       │
│  Locked formation z      │   │  vix.py — block / resume   │
│  Live β = hedge resize   │   │  earnings.py — blackout    │
│  only (if enabled)       │   │  and pre-earnings exit     │
└────────────┬─────────────┘   └─────────────┬──────────────┘
             └──────────────┬────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKTEST            src/backtest/                          │
│  engine.py     — daily loop; warmup load; keep last pairs   │
│  portfolio.py  — NAV, beta_hedge, drawdown                  │
│  costs.py      — commission, slippage, bid-ask, borrow      │
│  execution.py  — simulated fills                            │
└────────────────────────────┬────────────────────────────────┘
                             │  trade_log, nav_series,
                             │  pair_daily_mtm, blocked_entries
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  METRICS / REPORT    src/metrics/ + script 04               │
│  NAV/drawdown, monthly heatmap, exit mix, blocked entries   │
└─────────────────────────────────────────────────────────────┘
```

`src/tiering/` exists on disk and is **not** called. There is no
`bollinger.py`. Defaults are in `src/config.py`, not in this diagram.

---

## 3. Data flow

Script 01 writes `data/raw/` (prices, fundamentals, regime, earnings).
Script 02 writes `data/processed/` (returns, universe history). Script 03
writes `outputs/backtest_results/`. Script 04 writes `outputs/report/`.

Column schemas, fetch fallbacks, and known data issues:
[`data.md`](data.md). Output paths: [`file-structure.md`](file-structure.md).

There is no `pair_pnl.csv` or `walkforward_results.csv`.

---

## 4. Data contracts

Do not change these without updating `CLAUDE.md` / `AGENTS.md`.

```python
# ── Data layer ────────────────────────────────────────────────────────────

def load_returns(start: date | None, end: date | None) -> pd.DataFrame:
    """
    Columns : [date, ticker, log_return]
    Index   : RangeIndex
    Sorted  : date ascending
    NaN     : none
    start/end None → no bound on that side (engine uses this for warmup).
    """

def load_prices(start: date | None, end: date | None) -> pd.DataFrame:
    """Columns : [date, ticker, adj_close, volume]"""

def load_vix(start: date | None, end: date | None) -> pd.Series:
    """Index date, values VIX close, name 'vix'."""

def load_universe(as_of: date) -> list[str]:
    """Passing tickers on the latest reconstitution on or before as_of."""

def load_earnings_dates() -> pd.DataFrame:
    """
    Columns : [ticker, earnings_date] (tz-naive normalized Timestamps).
    Empty frame (with columns) when data/raw/earnings.parquet is missing —
    callers treat that as earnings features disabled.
    """

# ── Clustering and scoring ────────────────────────────────────────────────

def run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]:
    """cluster_id → tickers."""

def score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Columns:
        ticker_a, ticker_b, cluster_id, composite_score,
        beta_formation, mean_formation, std_formation, halflife_value
    Rows: at most finalists_per_cluster per cluster after
          half-life gate, min_cointegration_score, min_composite_score,
          and min_formation_beta.
    Empty frame keeps the same columns (engine must not clear active_pairs).
    """

# ── Signals and backtest ──────────────────────────────────────────────────

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
    LONG_SPREAD | SHORT_SPREAD | TAKE_PROFIT | STOP_LOSS |
    PLATEAU_STOP | TIME_STOP | HOLD
    z uses locked formation β/μ/σ, never the live hedge β.
    Entries require |z| within [entry_zscore, entry_zscore_max].
    """

def run_backtest(
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (trade_log, nav_series, pair_daily_mtm, blocked_entries).

    blocked_entries: [date, ticker_a, ticker_b, signal, reason]
    Exit actions include PLATEAU_STOP, EARNINGS_EXIT, and DOLLAR_STOP.
    """
```

---

## 5. Engine cadence

| Event | Cadence | Notes |
|---|---|---|
| Load prices / returns / VIX | Once | `start=None` so warmup history is available |
| Simulation dates | Daily | Only `[backtest_start_date, backtest_end_date]` |
| Universe | `universe_refresh_days` | `load_universe(as_of)` |
| Cluster + score | `clustering_refresh_days` | Same-day entries allowed |
| Empty score | — | Keep previous `active_pairs` |
| VIX / earnings | Daily | Block new entries; optional pre-earnings exit |
| Signal + execute | Daily | Formation z; `beta_hedge` for resize if enabled |
| Stop cooldown | `pair_stop_cooldown_days` | Per pair after `STOP_LOSS` / `PLATEAU_STOP` |

Pipeline order (01 → 04): [`README.md`](../README.md#pipeline).

---

## 6. Known architectural limitations

- Survivorship bias in yfinance listings.
- Fundamentals parquet is a current snapshot and is unused by scoring.
- Single-threaded daily loop.
- No broker; MOC fills; flat 2% borrow.
- Script 04 charts the full-sample CSVs; it is not a rolling re-fit.
- `src/tiering/`, `src/scrap/`, `working_model/`, and `scratch/` are not
  on the live path.
