# AGENTS.md — ARQ Pairs Trading

This file is read by Codex at the start of every session. It provides
persistent context about the project so you don't have to re-explain the
codebase each time. Keep this file updated as the project evolves.

---

## Project Overview

Systematic market-neutral pairs trading on a multi-sector S&P-style universe
(~95 names in `CANDIDATE_TICKERS`). K-means on a correlation distance matrix,
five-component composite score, then mean-reversion signals from a **locked
formation z-score** (not empirical percentiles).

**Live knobs:** `src/config.py`
**Implemented design:** `docs/architecture.md`
**Original v2.0 spec (not all live):** `docs/strategy.md`
**Issues / backtest results:** `docs/diagnostics.md`
**Canonical tree:** `docs/file-structure.md`
**Data sources:** `docs/data.md`
**Decisions / open questions:** `docs/decisions.md`, `docs/open-questions.md`

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pandas | 2.2+ | DataFrames, time series |
| numpy | 1.26+ | Numerical computation |
| scikit-learn | 1.4+ | K-means, silhouette scoring |
| statsmodels | 0.14+ | Johansen, OLS regression |
| yfinance | 0.2.40+ | Price and fundamental data |
| pyarrow | 15+ | Parquet I/O |
| matplotlib + seaborn | latest | Visualization (unused until script 05) |
| pytest | 8+ | Unit testing |
| ruff | 0.4+ | Linting and import sorting |
| uv | latest | Dependency and environment management |

---

## Repository Structure

```
arq-pairs-trading/
├── AGENTS.md / CLAUDE.md
├── README.md
├── pyproject.toml
│
├── docs/                 ← architecture (live), strategy (spec), diagnostics
├── data/
│   ├── sector_map.py     ← COMMITTED ticker → sector
│   ├── raw/              ← GITIGNORED
│   └── processed/        ← GITIGNORED
│
├── src/
│   ├── config.py
│   ├── data/             ← fetch.py, clean.py, load.py
│   ├── universe/         ← filter.py
│   ├── clustering/       ← correlation.py, kmeans.py
│   ├── scoring/          ← 5 components + composite.py
│   ├── signals/          ← hedge_ratio.py, spread.py, entry_exit.py
│   ├── regime/           ← vix.py, earnings.py
│   ├── backtest/         ← engine.py, portfolio.py, costs.py, execution.py
│   ├── metrics/          ← STUBS
│   ├── tiering/          ← leftover; engine does not call it
│   └── scrap/            ← prototypes; not the pipeline
│
├── scripts/              ← 01 → 02 → 03 → 04 (05 is a stub)
├── tests/
├── outputs/              ← GITIGNORED
├── working_model/        ← old prototype
└── scratch/              ← ad-hoc analysis
```

---

## Module Ownership

| Owner | Modules |
|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py`, `data/sector_map.py`, `scripts/01`, `scripts/02`, `tests/fixtures/` |
| Althan | `src/clustering/`, `src/scoring/` |
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

`start` / `end` may be `None` to drop that bound (warmup load).

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
load_returns(start: date | None, end: date | None) -> pd.DataFrame
    # columns: [date, ticker, log_return]
    # RangeIndex, sorted by date ascending, no NaN
    # None start/end = unbounded on that side

load_prices(start: date | None, end: date | None) -> pd.DataFrame
    # columns: [date, ticker, adj_close, volume]

load_vix(start: date | None, end: date | None) -> pd.Series
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

# src/scoring/candidate_pairs.py — Althan's output, consumed by composite
build_candidate_pairs(clusters: dict[int, list[str]]) -> pd.DataFrame
    # columns: [ticker_a, ticker_b, cluster_id]

# src/scoring/fundamentals.py — sector compatibility scorer
score_candidate_pairs(candidate_pairs: pd.DataFrame) -> pd.DataFrame
    # No returns or prices argument — reads from data/sector_map.py internally
    # Adds fundamentals_score column: 1.0 same sector, 0.4 cross-sector, 0.5 unknown

# src/scoring/composite.py — final scoring output
score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame
    # columns: [ticker_a, ticker_b, cluster_id, composite_score,
    #           beta_formation, mean_formation, std_formation, halflife_value]
    # exactly 1 pair per cluster, only pairs scoring >= CONFIG.min_composite_score
    # after half-life hard gate, min_cointegration_score, and β_F > min_formation_beta
    # raw absolute scores — no normalization
    # ticker_a is canonical stock A based on halflife direction
    # empty result must keep these columns; engine must not clear active_pairs

# src/signals/entry_exit.py — Anvay's output, consumed by backtest engine
get_signal(
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
) -> str
    # returns one of:
    # "LONG_SPREAD", "SHORT_SPREAD", "TAKE_PROFIT",
    # "STOP_LOSS", "TIME_STOP", "HOLD"
    # z uses locked formation β/μ/σ only

# src/backtest/engine.py
run_backtest(config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    # (trade_log, nav_series, pair_daily_mtm)
```

---

## Pipeline Execution Order

```
scripts/01_fetch_data.py      ← Run once at project start
scripts/02_build_universe.py  ← Run after 01, or after config changes
scripts/03_run_backtest.py    ← Full-calendar simulation
scripts/04_walkforward.py     ← Slices last oos_fraction of 03 CSVs (no re-fit)
scripts/05_generate_report.py ← Stub — do not expect charts
```

```bash
uv run python scripts/01_fetch_data.py --disable-proxy
uv run python scripts/02_build_universe.py
uv run python scripts/03_run_backtest.py
```

---

## Key Parameters (from src/config.py)

| Parameter | Default | Notes |
|---|---|---|
| `clustering_window` | 120 days | Correlation matrix for K-means |
| `formation_window` | 120 days | Locked β / μ / σ for z-score |
| `signal_window` | 60 days | Live hedge β for share resize only — not for z |
| `johansen_window` | 252 days | Johansen lookback |
| `k_min` / `k_max` | 4 / 6 | Silhouette-scored k |
| `entry_zscore` | 1.5 | Absolute formation z to enter |
| `take_profit_zscore` | 0.5 | Absolute z to take profit |
| `stop_loss_zscore` | 3.5 | Absolute z to stop |
| `time_stop_days` | 50 | Force close (must be ≥ 2× `halflife_max`) |
| `momentum_window` / `momentum_threshold` | 14 / 0.15 | Block entries if either leg moved this much |
| `pair_stop_cooldown_days` | 20 | Trading days after STOP_LOSS |
| `johansen_threshold` | 0.10 | Mapping aid; not a BH kill switch |
| `min_cointegration_score` | 0.40 | Soft floor on `1 − p_adj` |
| `min_formation_beta` | 0.0 | Drop non-positive formation β |
| `halflife_min` / `halflife_max` | 5 / 20 | Hard gate |
| `vix_entry_block` / `vix_resume` | 28 / 25 | Resume must hold 5 days |
| `weight_correlation_stability` | 0.20 | |
| `weight_cointegration` | 0.25 | |
| `weight_halflife` | 0.30 | Binary gate + score |
| `weight_volatility` | 0.15 | |
| `weight_fundamentals` | 0.10 | Sector labels, not P/S |
| `min_composite_score` | 0.55 | Absolute floor after raw weighted sum |
| `finalists_per_cluster` | 1 | Exactly 1 pair per cluster |
| `same_sector_score` / `cross_sector_score` / `unknown_sector_score` | 1.0 / 0.4 / 0.5 | |
| `beta_rebalance_threshold` | 0.15 | Resize short leg; do **not** overwrite `beta_at_entry` |
| `min_dollar_volume` | 25,000,000 | 30-day ADV$ proxy |
| `backtest_start_date` / `backtest_end_date` | 2022-01-01 / 2024-12-31 | Simulation window only |
| `oos_fraction` | 0.30 | Script 04 slice |
| `initial_capital` | 100000 | |
| `random_seed` | 42 | K-means |
| `data_quality_window` / `max_missing_days` | 90 / 5 | `clean.py` |

---

## Testing

```bash
uv run pytest
uv run pytest tests/test_scoring.py
uv run pytest --cov=src --cov-report=term-missing
```

Tests use synthetic data. They never call yfinance or read from `data/`.
All tests should complete in under 30 seconds.

---

## Common Mistakes to Avoid

**Stale processed data** — If you change filter thresholds in `config.py`,
delete `data/processed/` and re-run `scripts/02_build_universe.py`.

**Formation vs live β** — z-score uses locked `beta_formation` / `mean_formation`
/ `std_formation`. Live `signal_window` β updates `Position.beta_hedge` for
share resize only. Never assign the new β onto `beta_at_entry`.

**Warmup load** — The engine must load prices/returns/VIX with `start=None`
(or the fetch start). Restrict the **loop** to `[backtest_start_date,
backtest_end_date]`. Loading only the backtest window starves Johansen and
delays the first cluster.

**Empty score does not flatten** — If `score_candidates` returns no rows,
keep the previous `active_pairs`. Do not clear the book.

**Do not treat Johansen as a BH p < 0.10 kill switch** — Score is `1 − p_adj`
with `min_cointegration_score`. Half-life remains the hard statistical gate.

**Cross-cluster pairs** — Only within-cluster pairs are evaluated.

**Raw prices vs log prices** — Cointegration, hedge ratio, and spread use
log prices. Raw prices are for volume and ADV filters.

**Look-ahead in rolling windows** — Use `.shift(1)` where needed. A rolling
window ending on day T must not include day T's close when that close is
not yet known for the signal.

**yfinance bot detection** — `impersonate="chrome"` in
`_configure_yfinance_runtime()` in `fetch.py`.

**Regime data may be synthetic** — If `regime.parquet` row count equals the
business-day count exactly, you may still be on synthetic VIX/SPY.

**No normalization in composite** — Raw weighted scores, then
`min_composite_score`.

**Fundamentals scorer takes no time arguments** —
`fundamentals.score_candidate_pairs(candidate_pairs)` only. Sector map,
not P/S.

**Tiering and Bollinger are not on the live path** — `src/tiering/` is
leftover. There is no `bollinger.py`. Do not import either in new code.

**Script 04 is a slice** — It does not re-cluster quarterly. Script 05 is
a stub.

**Percentile thresholds are gone** — Signals are z-score. Do not
reintroduce `entry_percentile_*` in new code unless config and tests
change together.

---

## Scope Cuts (if behind schedule)

If the team falls behind, cut in this order. Document every cut in
`docs/decisions.md`.

| If behind by | Cut | Never cut |
|---|---|---|
| 1 day | Earnings blackout filter | Transaction costs |
| 2 days | Walk-forward / OOS slice polish | Held-out OOS period |
| 3 days | Sector component (set weight to 0) | Formation-locked z / dollar-neutral sizing |
| 4+ days | Min composite threshold | Half-life hard gate |

---

## Git Conventions

Branch naming: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `data/`

```bash
git checkout -b feat/composite-scorer
```

Commit format: `type(scope): short summary`

```bash
git commit -m "feat(scoring): add half-life scorer with AR(1) regression"
git commit -m "fix(data): forward-fill missing days up to 1 day only"
git commit -m "config: raise johansen threshold from 0.05 to 0.10"
```

Full conventions: `docs/git-conventions.md`

---

## Questions and Decisions

Before making a non-obvious design decision, check `docs/open-questions.md`
to see if it is already being discussed. After deciding, log it in
`docs/decisions.md` so the team doesn't revisit it.

If Codex makes a mistake — wrong window, wrong column name, wrong interface
— add a correction to this file under a new "Corrections" section so it
does not repeat the error in future sessions.
