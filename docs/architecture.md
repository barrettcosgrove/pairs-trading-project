# Architecture

# ARQ Pairs Trading — System Architecture

This document describes the system design of the ARQ pairs trading pipeline:
how components are organized, how data flows between them, what technology
is used and why, and what the agreed interfaces between modules are.

If code and this document disagree, fix the code.

---

## 1. Design Philosophy

The pipeline is built around three principles:

**Pipeline pattern over monolith.** The strategy is a series of
transformations: raw data → cleaned → universe → clusters → scored pairs →
confirmed pairs → signals → trades → metrics. Each transformation is a
discrete module with a clear input and output. Modules are independently
testable, independently ownable, and independently replaceable.

**Cache intermediate outputs.** Every pipeline stage writes its output to
disk as a parquet file before the next stage reads it. This means you can
change the scoring module and re-run from that stage forward without
re-fetching data or re-clustering. It also means you can inspect any
intermediate output in a notebook to debug unexpected behavior.

**Centralize configuration.** Every tunable parameter lives in
`src/config.py` and nowhere else. No magic numbers in module files. This
makes sensitivity analysis trivial — swap in a different config and re-run.

---

## 2. Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL DATA SOURCES                                      │
│  yfinance (prices, fundamentals, VIX, SPY)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER          src/data/                              │
│                                                             │
│  fetch.py     — Downloads raw data, writes data/raw/        │
│  clean.py     — Validates, forward-fills, drops bad tickers │
│  load.py      — Typed read-only interface for all modules   │
└────────────────────────────┬────────────────────────────────┘
                             │  load_returns(), load_prices(),
                             │  load_vix(), load_universe()
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIVERSE LAYER      src/universe/                          │
│                                                             │
│  filter.py    — Applies 5 hard pre-filters monthly          │
│               — Writes universe_history.parquet             │
└────────────────────────────┬────────────────────────────────┘
                             │  list[str] of passing tickers
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  CLUSTERING LAYER    src/clustering/                        │
│                                                             │
│  correlation.py — Builds 120-day correlation distance matrix│
│                 — Drops incomplete ticker histories         │
│                 — Saves / loads monthly snapshots           │
│  kmeans.py      — Silhouette-scored K-means (k ∈ [4, 6])   │
│                 — Returns cluster assignments               │
└────────────────────────────┬────────────────────────────────┘
                             │  dict[cluster_id, list[ticker]]
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  SCORING LAYER       src/scoring/                           │
│                                                             │
│  candidate_pairs.py        — Expands clusters into pairs    │
│  correlation_stability.py  — 20% weight                     │
│  cointegration.py          — 30% weight (Johansen + BH FDR) │
│  halflife.py               — 25% weight (AR(1) regression)  │
│  volatility.py             — 15% weight (dual window)       │
│  fundamentals.py           — 10% weight (P/S, revenue)      │
│  composite.py              — Absolute scores, applies       │
│                            — threshold, 1 pair per cluster  │
└────────────────────────────┬────────────────────────────────┘
                             │  DataFrame[ticker_a, ticker_b,
                             │            cluster_id, score]
                             ▼
┌──────────────────────────────────────┐  ┌────────────────────────────────────┐
│  SIGNAL LAYER  src/signals/          │  │  REGIME LAYER  src/regime/         │
│                                      │  │                                    │
│  hedge_ratio.py — Rolling 60d OLS    │  │  vix.py       — Portfolio filter   │
│  spread.py      — Log price spread   │  │  earnings.py  — Blackout windows   │
│  entry_exit.py  — Empirical %ile     │  │                                    │
│                   thresholds         │  │                                    │
└──────────────────┬───────────────────┘  └────────────────┬───────────────────┘
                   │  signal: str                          │  permitted: bool
                   └──────────────────┬────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│  BACKTEST LAYER      src/backtest/                          │
│                                                             │
│  engine.py     — Daily simulation loop                      │
│  portfolio.py  — Position tracking, NAV, tier pools         │
│  costs.py      — Commission, slippage, borrow               │
│  execution.py  — Simulated fills, order sequencing          │
└────────────────────────────┬────────────────────────────────┘
                             │  trade_log, nav_series
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  METRICS LAYER       src/metrics/                           │
│                                                             │
│  performance.py — Sharpe, Sortino, drawdown, win rate       │
│  reporting.py   — Charts and tables for final report        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Tool | Version | Purpose | Why This Choice |
|---|---|---|---|
| Python | 3.11+ | Runtime | f-strings, better type hints, faster than 3.10 |
| pandas | 2.2+ | DataFrames, time series | Standard for financial data manipulation |
| numpy | 1.26+ | Numerical computation | Underlies pandas, required for statsmodels |
| scikit-learn | 1.4+ | K-means, silhouette scoring | Battle-tested, consistent API |
| statsmodels | 0.14+ | Johansen, KPSS, OLS | Best Python library for these specific tests |
| yfinance | 0.2.40+ | Price and fundamental data | Free, no API key, sufficient for project scope |
| curl_cffi | latest | Proxy-bypass session for yfinance | Provides a `trust_env=False` session that bypasses OS and env-var proxy settings when `--disable-proxy` is used |
| pyarrow | 15+ | Parquet I/O | Fast columnar storage, preserves dtypes |
| matplotlib + seaborn | latest | Visualization | Standard plotting, sufficient for report charts |
| pytest | 8+ | Unit testing | Industry standard, simple fixture system |
| ruff | 0.4+ | Linting and import sorting | Replaces flake8 + isort in one fast tool |
| uv | latest | Dependency management | Fast, modern replacement for pip + venv |

### Why a Custom Backtest Engine

The strategy was built using a custom loop-based engine in `src/backtest/`
rather than a third-party library like vectorbt or backtrader.

Reasons:
- **Full control**: tier pool mechanics, deferred resizing, and dollar-neutral
  rebalancing are non-standard and difficult to express in library abstractions
- **Debuggability**: a plain Python loop is easy to step through in a debugger
- **Timeline**: learning vectorbt's API in a two-week sprint would cost more
  time than writing 400 lines of custom simulation code
- **14 pairs maximum**: performance is not a bottleneck at this scale

---

## 4. Directory Structure

```
arq-pairs-trading/
│
├── src/                    ← All production Python code
│   ├── config.py           ← Single source of truth for all parameters
│   ├── data/               ← Data acquisition and access
│   ├── universe/           ← Universe filtering
│   ├── clustering/         ← Correlation matrix and K-means
│   ├── scoring/            ← Five component scorers + aggregator
│   ├── signals/            ← Hedge ratio, spread, entry/exit signals
│   ├── regime/             ← VIX, earnings filters
│   ├── backtest/           ← Simulation engine
│   └── metrics/            ← Performance stats and reporting
│
├── scripts/                ← Numbered orchestration scripts (run in order)
├── tests/                  ← pytest unit tests with synthetic fixtures
├── docs/                   ← Strategy and architecture documentation
├── data/                   ← Data storage (raw/ and processed/ gitignored)
└── outputs/                ← All generated results (gitignored)
```

---

## 5. Data Flow and Storage

### Raw Data (written once, never modified)

```
data/raw/prices.parquet
    ticker: str, date: datetime64, open: float64, high: float64,
    low: float64, close: float64, adj_close: float64, volume: int64
    Primary key: (ticker, date)

data/raw/fundamentals.parquet
    ticker: str, fetch_date: date,
    price_to_sales: float64, revenue_growth_ttm: float64

data/raw/regime.parquet
    date: datetime64, vix: float64, spy: float64
```

### Processed Data (regenerated by scripts/02_build_universe.py)

```
data/processed/returns.parquet
    ticker: str, date: datetime64, log_return: float64
    No NaN values. Only tickers passing universe filters.

data/processed/universe_history.parquet
    date: datetime64, ticker: str, passed_filters: bool
    One row per ticker per monthly reconstitution date.

data/processed/correlation_matrices/YYYY-MM.parquet
    NxN DataFrame where index and columns are ticker strings.
    Values are distance (1 - correlation), range [0, 2].
    One file per month, cached to avoid recomputation and reloaded
    by the clustering layer when available.
```

### Outputs (generated by backtest and reporting scripts)

```
outputs/backtest_results/trade_log.csv
outputs/backtest_results/nav_series.csv
outputs/backtest_results/pair_pnl.csv
outputs/backtest_results/walkforward_results.csv
outputs/report/*.png
outputs/data_quality_report.txt
```

---

## 6. Module Ownership

| Owner | Modules | Key Interfaces Produced |
|---|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py` | `load_returns()`, `load_prices()`, `load_vix()`, `load_universe()` |
| Althan | `src/clustering/`, `src/scoring/` | `run_clustering()`, `score_candidates()` |
| Anvay | `src/signals/`, `src/regime/`, `src/backtest/` | `get_signal()`, `run_backtest()` |
| Nanshu | `src/metrics/`, `scripts/03-05` | `compute_metrics()`, all report outputs |

Cross-module changes require a PR reviewed by the other owner.

---

## 7. Data Contracts

These are the agreed function signatures between modules. Do not change
these without updating CLAUDE.md and notifying the affected owner.

```python
# ── Barrett → Everyone ───────────────────────────────────────────────────

def load_returns(start: date, end: date) -> pd.DataFrame:
    """
    Columns : [date, ticker, log_return]
    Index   : RangeIndex (not date)
    Sorted  : by date ascending
    NaN     : none — dropped during cleaning
    """

def load_prices(start: date, end: date) -> pd.DataFrame:
    """
    Columns : [date, ticker, adj_close, volume]
    """

def load_vix(start: date, end: date) -> pd.Series:
    """
    Index  : date (datetime64)
    Values : VIX daily close (float64)
    Name   : "vix"
    """

def load_universe(as_of: date) -> list[str]:
    """
    Returns tickers passing all hard filters on the given date.
    Uses the most recent monthly reconstitution on or before as_of.
    """

# ── Althan → Anvay ───────────────────────────────────────────────────────

def run_clustering(
    distance_matrix: pd.DataFrame,
) -> dict[int, list[str]]:
    """
    Keys   : cluster id (int, 0-indexed)
    Values : list of ticker strings assigned to that cluster
    """

def score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame:
    """
    Columns : [ticker_a, ticker_b, cluster_id, composite_score]
    Rows    : top 1 pair per cluster passing min_composite_score
    Sorted  : by composite_score descending
    """

# ── Anvay → Nanshu ───────────────────────────────────────────────────────

def get_signal(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date,
) -> str:
    """
    Returns one of:
        "LONG_SPREAD"  — long A, short B
        "SHORT_SPREAD" — short A, long B
        "TAKE_PROFIT"  — close position
        "STOP_LOSS"    — close position immediately
        "TIME_STOP"    — close position (20-day limit reached)
        "HOLD"         — no action
    """

def run_backtest(
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns : (trade_log, nav_series)

    trade_log columns:
        date, ticker_a, ticker_b, action, shares_a, shares_b,
        price_a, price_b, cost, pnl, tier

    nav_series columns:
        date, nav, cash, gross_exposure, drawdown_from_peak
    """
```

---

## 8. Pipeline Execution Order

Run scripts in this exact order. Each depends on the outputs of the previous.

```bash
# Step 1 — Download all raw data (run once at project start)
# --disable-proxy bypasses OS and env-var proxy settings that block Yahoo Finance
uv run python scripts/01_fetch_data.py --disable-proxy

# If Yahoo rate-limits, fetch regime and fundamentals in separate passes:
#   uv run python scripts/01_fetch_data.py --disable-proxy --stage regime
#   uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals --resume

# Step 2 — Clean data and build universe history
uv run python scripts/02_build_universe.py

# Step 3 — Run full backtest on training period
uv run python scripts/03_run_backtest.py

# Step 4 — Walk-forward validation
uv run python scripts/04_walkforward.py

# Step 5 — Generate all charts and tables
uv run python scripts/05_generate_report.py
```

To re-run from a specific stage (e.g. after changing scoring weights):

```bash
# No need to re-fetch — start from backtest
uv run python scripts/03_run_backtest.py
```

To reset everything and start from scratch:

```bash
rm -rf data/raw/ data/processed/ outputs/
uv run python scripts/01_fetch_data.py
uv run python scripts/02_build_universe.py
uv run python scripts/03_run_backtest.py
```

---

## 9. Testing Strategy

Unit tests live in `tests/` and mirror the `src/` structure. All tests use
synthetic data from `tests/fixtures/` — no network calls, no real parquet
files. The full suite runs in under 30 seconds.

```bash
uv run pytest                              # full suite
uv run pytest tests/test_scoring.py       # single module
uv run pytest --cov=src                   # with coverage
```

Each module owner writes tests for their own module. Tests must pass before
any PR is merged into develop.

**What is tested:**
- Correct output shapes and column names from each module
- Boundary conditions (empty cluster, all pairs failing filters)
- Dollar-neutrality enforcement in portfolio manager
- Cost deduction on every trade in backtest engine
- No look-ahead bias in rolling window calculations

**What is not tested here:**
- End-to-end backtest correctness — validated manually by running
  `scripts/03_run_backtest.py` on a short date range and inspecting outputs
- Real data quality — validated by reviewing `outputs/data_quality_report.txt`
  after each fetch

---

## 10. Known Architectural Limitations

**Survivorship bias in universe construction.**
yfinance only returns currently-listed stocks. Stocks delisted or acquired
during the backtest window are absent from the universe. This overstates
backtest performance. Documented in `docs/data.md`.

**Fundamental data is not point-in-time.**
P/S ratios and revenue growth are fetched as a current snapshot and held
constant over the backtest. A production system would use as-reported data
with earnings release date lags. Documented as known simplification.

**Single-threaded backtest engine.**
The daily simulation loop is sequential. For 14 pairs over 2.5 years this
runs in under 15 minutes, which is acceptable for the project scope. A
production system would vectorize the inner loop or use multiprocessing.

**No broker integration.**
The execution module simulates fills using market-on-close prices. There
is no connection to a live broker. Short borrow costs use a flat 2%
annual assumption rather than per-stock rates.