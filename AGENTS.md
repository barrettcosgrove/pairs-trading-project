# AGENTS.md — ARQ Pairs Trading

This file is read by Codex at the start of every session. It provides
persistent context about the project so you don't have to re-explain the
codebase each time. Keep this file updated as the project evolves.

---

## Project Overview

Systematic market-neutral pairs trading strategy on US technology stocks.
The pipeline identifies cointegrated stock pairs using K-means clustering on
a correlation distance matrix, scores candidates with a five-component
composite score, filters finalists by a minimum score threshold, and generates
mean-reversion entry/exit signals using empirical spread percentiles.

**Full strategy specification:** `docs/strategy.md`
**System design and data flow:** `docs/architecture.md`
**Data sources and schemas:** `docs/data.md`
**Build roadmap and ownership:** `docs/implementation-plan.md`
**Design decisions log:** `docs/decisions.md`
**Unresolved questions:** `docs/open-questions.md`

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pandas | 2.2+ | DataFrames, time series |
| numpy | 1.26+ | Numerical computation |
| scikit-learn | 1.4+ | K-means, silhouette scoring |
| statsmodels | 0.14+ | Johansen, KPSS, OLS regression |
| yfinance | 0.2.40+ | Price and fundamental data |
| pyarrow | 15+ | Parquet I/O |
| matplotlib + seaborn | latest | Visualization |
| pytest | 8+ | Unit testing |
| ruff | 0.4+ | Linting and import sorting |
| uv | latest | Dependency and environment management |

---

## Repository Structure

```
arq-pairs-trading/
├── AGENTS.md                    ← This file
├── README.md                    ← Quick start for humans
├── pyproject.toml               ← Dependencies (managed by uv)
├── .gitignore
├── .env.example
│
├── docs/                        ← Strategy and architecture docs
├── data/
│   ├── sector_map.py            ← COMMITTED: ticker → subsector mapping
│   ├── raw/                     ← GITIGNORED: fetched from yfinance
│   └── processed/               ← GITIGNORED: derived by pipeline
│
├── src/
│   ├── config.py                ← ALL parameters live here, nowhere else
│   ├── data/                    ← fetch.py, clean.py, load.py
│   ├── universe/                ← filter.py
│   ├── clustering/              ← correlation.py, kmeans.py
│   ├── scoring/                 ← candidate_pairs.py + 5 component files + composite.py
│   ├── signals/                 ← hedge_ratio.py, spread.py, entry_exit.py
│   ├── regime/                  ← vix.py, earnings.py
│   ├── backtest/                ← engine.py, portfolio.py, costs.py, execution.py
│   └── metrics/                 ← performance.py, reporting.py
│
├── scripts/                     ← Run in order: 01 → 02 → 03 → 04 → 05
├── tests/                       ← pytest, uses fixtures/ not real data
└── outputs/                     ← GITIGNORED: all generated results
```

---

## Module Ownership

| Owner | Modules |
|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py`, `data/sector_map.py`, `scripts/01`, `scripts/02`, `tests/fixtures/` |
| Althan | `src/clustering/`, `src/scoring/`|
| Anvay | `src/signals/`, `src/regime/`, `src/backtest/` |
| Nanshu | `src/metrics/`, `scripts/03`, `scripts/04`, `scripts/05` |

When working on a module, check ownership before editing files outside your
area. Cross-module changes require a PR reviewed by the other owner.

---

## Critical Rules — Read Before Writing Any Code

### 1. Never hardcode parameters
Every tunable value lives in `src/config.py` as a frozen dataclass. Import
`CONFIG` and reference it. Never write magic numbers inline.

```python
# WRONG
window = 60
threshold = 0.10

# CORRECT
from src.config import CONFIG
window = CONFIG.signal_window
threshold = CONFIG.johansen_threshold
```

### 2. Never read parquet files directly
All data access goes through `src/data/load.py`. Never call `pd.read_parquet()`
in a module file.

```python
# WRONG
df = pd.read_parquet("data/processed/returns.parquet")

# CORRECT
from src.data.load import load_returns
df = load_returns(start=start_date, end=end_date)
```

### 3. Never call yfinance outside fetch.py
All yfinance calls are isolated to `src/data/fetch.py`. Other modules never
import yfinance directly.

### 4. fetch.py is the only file that imports yfinance or curl_cffi
Both `yfinance` and `curl_cffi` are isolated to `src/data/fetch.py`. No
other module imports either library. `curl_cffi` provides the proxy-bypass
session used by yfinance; treat it as an internal implementation detail of
the data fetcher.

### 5. All functions must have docstrings
Every function needs a one-line summary, Args, and Returns. This is how
teammates (and Codex) understand interfaces without reading implementations.


```python
def compute_halflife(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life of a spread via AR(1) regression.

    Args:
        spread: Time series of spread values, indexed by date.

    Returns:
        Half-life in trading days. Returns inf if slope is non-negative
        (diverging pair). Returns inf if regression fails.
    """
```

### 6. Use logging, not print
```python
# WRONG
print(f"Processing pair {ticker_a}/{ticker_b}")

# CORRECT
import logging
logger = logging.getLogger(__name__)
logger.info("Processing pair %s/%s", ticker_a, ticker_b)
```

### 7. No look-ahead bias
At every point in the backtest loop, only use data available on or before
that date. Rolling windows must be computed from past data only. When in
doubt, add a comment explaining why the data access is safe.

---

## Data Contracts

These are the agreed interfaces between modules. Do not change these
signatures without updating this file and notifying the affected owner.

```python
# src/data/load.py — Barrett's outputs, consumed by everyone
load_returns(start: date, end: date) -> pd.DataFrame
    # columns: [date, ticker, log_return]
    # indexed by date, sorted ascending
    # no NaN values

load_prices(start: date, end: date) -> pd.DataFrame
    # columns: [date, ticker, adj_close, volume]

load_vix(start: date, end: date) -> pd.Series
    # index: date, values: VIX close

load_universe(as_of: date) -> list[str]
    # tickers passing all hard filters on that date

# src/clustering/correlation.py — Althan's input/output
build_distance_matrix(returns: pd.DataFrame, window: int) -> pd.DataFrame
    # NxN DataFrame, index and columns are ticker strings
    # values are (1 - correlation), range [0, 2]

# src/clustering/kmeans.py — Althan's output, consumed by scoring
run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]
    # keys: cluster id (int)
    # values: list of ticker strings in that cluster

# src/scoring/candidate_pairs.py - Althan's output, consumed by composite
build_candidate_pairs(clusters: dict[int, list[str]]) -> pd.DataFrame
    # columns: [ticker_a, ticker_b, cluster_id]


# src/scoring/composite.py — Althan's output, consumed by backtest engine
score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date
) -> pd.DataFrame
    # columns: [ticker_a, ticker_b, cluster_id, composite_score]
    # sorted by composite_score descending within each cluster
    # top 1 pair per cluster passing min_composite score

# src/signals/entry_exit.py — Anvay's output, consumed by backtest engine
get_signal(
    ticker_a: str,
    ticker_b: str,
    prices: pd.DataFrame,
    as_of: date
) -> str
    # returns one of:
    # "LONG_SPREAD", "SHORT_SPREAD", "TAKE_PROFIT",
    # "STOP_LOSS", "TIME_STOP", "HOLD"
```

---

## Pipeline Execution Order

```
scripts/01_fetch_data.py      ← Run once at project start
scripts/02_build_universe.py  ← Run after 01, or after config changes
scripts/03_run_backtest.py    ← Main entry point
scripts/04_walkforward.py     ← Run after 03
scripts/05_generate_report.py ← Run after 03 and 04
```

Each script depends on the outputs of the previous one. Do not skip steps.

To run:
```bash
uv run python scripts/01_fetch_data.py --disable-proxy
uv run python scripts/02_build_universe.py
uv run python scripts/03_run_backtest.py
```

---

## Key Parameters (from src/config.py)

| Parameter | Default | Notes |
|---|---|---|
| `clustering_window` | 120 days | Correlation matrix window — decoupled from signal window |
| `signal_window` | 60 days | Window for β, μ, σ, empirical percentiles — must be same for all three |
| `entry_zscore` | 1.5 | Entry threshold (absolute Z-score of spread) |
| `take_profit_zscore` | 0.5 | Take-profit threshold (absolute Z-score reverts below this) |
| `stop_loss_zscore` | 3.5 | Stop-loss threshold (absolute Z-score stretches beyond this) |
| `johansen_threshold` | 0.10 | p-value threshold (BH-corrected) |
| `halflife_min` | 5 | Minimum acceptable half-life in days |
| `halflife_max` | 20 | Maximum acceptable half-life in days |
| `vix_entry_block` | 28 | VIX level that blocks all new entries |
| `vix_resume` | 25 | VIX level to resume (must hold 5 days) |
| `min_composite_score` | 0.55 | Minimum threshold for a cluster finalist to be traded |
| `max_weight_per_pair` | 1.0 | Maximum fraction of NAV allocated to a single pair |
| `min_formation_spread_std` | 1e-6 | Floor on σ_F in formation z-score denominator |
| `beta_rebalance_threshold` | 0.15 | Rebalance short leg when β drifts beyond this |
| `short_borrow_annual` | 0.02 | Flat borrow cost assumption |
| `random_seed` | 42 | Fixed K-means seed for reproducibility |
| `min_dollar_volume` | 25,000,000 | 30-day avg daily dollar volume proxy for market cap — replaces min_market_cap |
| `data_quality_window` | 90 | Trailing window (days) used to count missing adj_close values in clean.py |
| `max_missing_days` | 5 | Maximum missing days allowed in any trailing `data_quality_window` before a ticker is dropped |

---

## Testing

```bash
# Run full suite (must pass before any PR)
uv run pytest

# Run a specific module
uv run pytest tests/test_scoring.py

# With coverage
uv run pytest --cov=src --cov-report=term-missing
```

Tests use synthetic data from `tests/fixtures/`. They never call yfinance
or read from `data/`. All tests should complete in under 30 seconds.

---

## Common Mistakes to Avoid

**Stale processed data** — If you change filter thresholds in `config.py`,
delete `data/processed/` and re-run `scripts/02_build_universe.py`. Old
processed files will silently produce wrong results.

**Window inconsistency** — β, μ, and σ must all use `CONFIG.signal_window`.
If any one of them uses a different window, entry thresholds are
miscalibrated. This is a silent bug that produces plausible-looking but
wrong signals.

**Cross-cluster pairs** — Only within-cluster pairs are evaluated. Never
pass cross-cluster pairs to the scoring module.

**Raw prices vs. log prices** — Cointegration tests, hedge ratio estimation,
and spread calculation all use log prices. Raw prices are only used for
volume and ADV filters. Do not mix them.

**Look-ahead in rolling windows** — Use `.shift(1)` where needed. A rolling
window ending on day T must not include day T's data when generating day T's
signal.

**yfinance bot detection** — Always use `impersonate="chrome"` when
creating a curl_cffi Session in fetch.py. Without it, Yahoo detects
the request as automated and returns 429. This is set in
`_configure_yfinance_runtime()`.

**Regime data may be synthetic during development** — Replace with real fetched
data before the final backtest. Check `data/raw/regime.parquet` row count:
synthetic data will have exactly the number of business days in the date range;
real data may have slightly fewer due to market holidays. If the row count
matches the business-day count exactly, you are still on synthetic data.

---

## Scope Cuts (if behind schedule)

If the team falls behind, cut in this order. Document every cut in
`docs/decisions.md`.

| If behind by | Cut | Never cut |
|---|---|---|
| 1 day | Earnings blackout filter | Transaction costs |
| 2 days | Walk-forward validation | OOS held-out period |
| 3 days | Fundamental compatibility component | Dollar-neutral sizing |
| 4+ days | Walk-forward validation | Held-out OOS period |



---

## Git Conventions

Branch naming: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `data/`

```bash
# Example
git checkout -b feat/composite-scorer
```

Commit format: `type(scope): short summary`

```bash
# Examples
git commit -m "feat(scoring): add half-life scorer with AR(1) regression"
git commit -m "fix(data): forward-fill missing days up to 1 day only"
git commit -m "config: raise johansen threshold from 0.05 to 0.10"
```

Full conventions: `docs/git_conventions.md`

---

## Questions and Decisions

Before making a non-obvious design decision, check `docs/open-questions.md`
to see if it is already being discussed. After deciding, log it in
`docs/decisions.md` so the team doesn't revisit it.

If Codex makes a mistake — wrong window, wrong column name, wrong interface
— add a correction to this file under a new "Corrections" section so it
does not repeat the error in future sessions.