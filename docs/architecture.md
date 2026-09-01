# ARQ Pairs Trading — System Architecture

This document describes the **implemented** pipeline: modules, data flow,
and contracts. Live parameters are in `src/config.py`. The original product
spec is `docs/strategy.md` (several sections there — percentiles, tiering,
Bollinger — are not what the engine does). Empirical issues:
`docs/diagnostics.md`.

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
│  yfinance (prices, fundamentals)                            │
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
│  correlation.py — 120-day D = 1 − ρ                         │
│  kmeans.py      — silhouette k ∈ [4, 6]                     │
│  Cadence: CONFIG.clustering_refresh_days (63)               │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SCORING             src/scoring/                           │
│  corr 0.20 / coint 0.25 / half-life 0.30 / vol 0.15 /       │
│  sector 0.10                                                │
│  Hard gate: half-life in [5, 20]                            │
│  Soft floor: min_cointegration_score 0.40; β_F > 0          │
│  Top 1 pair / cluster if composite ≥ 0.55                   │
└────────────────────────────┬────────────────────────────────┘
                             │  ticker_a, ticker_b, cluster_id,
                             │  composite_score, beta/mean/std_formation,
                             │  halflife_value
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  SIGNALS  src/signals/   │   │  REGIME  src/regime/       │
│  Locked formation z      │   │  vix.py — block > 28       │
│  Live 60d β = hedge only │   │  earnings.py — quarter-end │
│  entry 1.5 / TP 0.5 /    │   │                            │
│  SL 3.5 / time 50d       │   │                            │
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
                             │  trade_log, nav_series, pair_daily_mtm, blocked_entries
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  METRICS / REPORT    src/metrics/ + script 04               │
│  NAV/drawdown, monthly heatmap, exit mix, blocked entries   │
└─────────────────────────────────────────────────────────────┘
```

`src/tiering/` exists on disk and is **not** called. There is no
`bollinger.py`.

---

## 3. Technology Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pandas | 2.2+ | DataFrames, time series |
| numpy | 1.26+ | Numerical computation |
| scikit-learn | 1.4+ | K-means, silhouette |
| statsmodels | 0.14+ | Johansen, OLS |
| yfinance | 0.2.40+ | Prices and fundamentals |
| curl_cffi | latest | Chrome-impersonated session in `fetch.py` |
| pyarrow | 15+ | Parquet I/O |
| matplotlib + seaborn | latest | Report charts (script 04) |
| pytest | 8+ | Unit tests |
| ruff | 0.4+ | Lint |
| uv | latest | Environment |

The backtest is a custom daily loop (not vectorbt / backtrader) so formation
locks, VIX hysteresis, and dollar-neutral β resize stay explicit.

---

## 4. Directory Structure

See [`file-structure.md`](file-structure.md) for the full tree and
[`project-structure.md`](project-structure.md) for file-by-file notes.

```
src/          production modules (tiering/ and scrap/ are not live)
scripts/      01 fetch → 02 universe → 03 backtest → 04 report
tests/        synthetic pytest
docs/         strategy (spec), architecture (live), diagnostics (results)
data/         sector_map.py + gitignored raw/processed
outputs/      gitignored CSVs
```

---

## 5. Data Flow and Storage

### Raw (written by script 01)

```
data/raw/prices.parquet
    ticker, date, open, high, low, close, adj_close, volume
    Fetch window: 2019-01-01 → 2026-04-01. ~95 CANDIDATE_TICKERS.

data/raw/fundamentals.parquet
    ticker, fetch_date, price_to_sales, revenue_growth_ttm
    Fetched but not used by live scoring.

data/raw/regime.parquet
    date, vix, spy
    VIX: CBOE then yfinance. SPY: Alpha Vantage if keyed, else yfinance,
    else synthetic (must not ship a final backtest on synthetic SPY/VIX).
```

### Processed (script 02)

```
data/processed/returns.parquet
    ticker, date, log_return

data/processed/universe_history.parquet
    date, ticker, passed_filters

data/processed/correlation_matrices/YYYY-MM.parquet
    Optional cache from correlation.py (NxN distance).
```

### Outputs

```
outputs/backtest_results/trade_log.csv          # script 03
outputs/backtest_results/nav_series.csv         # script 03
outputs/backtest_results/pair_daily_mtm.csv     # script 03
outputs/backtest_results/blocked_entries.csv    # script 03
outputs/data_quality_report.txt                 # clean.py / script 02
outputs/report/                                 # script 04 charts + metrics_summary.txt
```

There is no `pair_pnl.csv` or `walkforward_results.csv`.

---

## 6. Module Ownership

| Owner | Modules | Interfaces |
|---|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py`, `data/sector_map.py`, scripts 01–02 | `load_*` |
| Althan | `src/clustering/`, `src/scoring/` | `run_clustering()`, `score_candidates()` |
| Anvay | `src/signals/`, `src/regime/`, `src/backtest/` | `get_signal()`, `run_backtest()` |
| Nanshu | `src/metrics/`, scripts 03–04 | CSV writers; report charts |

Cross-module changes need a PR reviewed by the other owner.

---

## 7. Data Contracts

Do not change these without updating `CLAUDE.md` / `AGENTS.md` and notifying
the owner.

```python
# ── Barrett → everyone ────────────────────────────────────────────────────

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

# ── Althan → Anvay ────────────────────────────────────────────────────────

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
    Rows: at most finalists_per_cluster (1) per cluster after
          half-life gate, min_cointegration_score, min_composite_score,
          and min_formation_beta.
    Empty frame keeps the same columns (engine must not clear active_pairs).
    """

# ── Anvay → engine / Nanshu ───────────────────────────────────────────────

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
    LONG_SPREAD | SHORT_SPREAD | TAKE_PROFIT | STOP_LOSS | TIME_STOP | HOLD
    z uses locked formation β/μ/σ, never the live hedge β.
    """

def run_backtest(
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (trade_log, nav_series, pair_daily_mtm).

    trade_log:
        date, ticker_a, ticker_b, action, shares_a, shares_b,
        price_a, price_b, cost, pnl,
        dollar_allocation_at_entry, realized_net_usd, return_pct

    nav_series:
        date, nav, cash, gross_exposure, drawdown_from_peak
        (NAV snapshot is pre-trade in the day loop.)

    pair_daily_mtm:
        date, ticker_a, ticker_b, direction, cluster_id, mtm_usd,
        gross_exposure_pair, dollar_allocation_at_entry,
        portfolio_nav_post_trade
    """
```

---

## 8. Engine cadence

| Event | Cadence | Notes |
|---|---|---|
| Load prices / returns / VIX | Once | `start=None` so 2019+ warmup is available |
| Simulation dates | Daily | Only `[backtest_start_date, backtest_end_date]` |
| Universe | `universe_refresh_days` (21) | `load_universe(as_of)` |
| Cluster + score | `clustering_refresh_days` (63) | Same-day entries allowed |
| Empty score | — | Keep previous `active_pairs` |
| VIX / earnings | Daily | Block new entries only |
| Signal + execute | Daily | Formation z; `beta_hedge` for resize |
| Stop cooldown | 20 trading days | Per pair after `STOP_LOSS` |

---

## 9. Pipeline execution order

```bash
uv run python scripts/01_fetch_data.py --disable-proxy
# optional: --stage prices|regime|fundamentals --resume

uv run python scripts/02_build_universe.py
uv run python scripts/03_run_backtest.py
uv run python scripts/04_generate_report.py
```

Script 04 reads script 03 CSVs and writes `outputs/report/`. It does not
re-run the engine or slice an OOS window.

To reset caches:

```bash
rm -rf data/raw/ data/processed/ outputs/
uv run python scripts/01_fetch_data.py --disable-proxy
uv run python scripts/02_build_universe.py
uv run python scripts/03_run_backtest.py
```

---

## 10. Testing

```bash
uv run pytest
uv run pytest tests/test_scoring.py
uv run pytest --cov=src
```

Synthetic only. Covered areas include clustering, the five scorers,
composite gates, signals, and backtest portfolio helpers. End-to-end
correctness is checked by running script 03 and reading
`outputs/backtest_results/` plus `docs/diagnostics.md`.

---

## 11. Known architectural limitations

- Survivorship bias in yfinance listings.
- Fundamentals parquet is a current snapshot and is unused by scoring.
- Single-threaded daily loop.
- No broker; MOC fills; flat 2% borrow.
- Script 04 is a calendar slice, not a rolling re-fit.
- Metrics / report layer is unimplemented.
- `src/tiering/`, `src/scrap/`, `working_model/`, and `scratch/` are not
  on the live path.
