# AGENTS.md — Pairs Trading

This file is read by Codex at the start of every session. It provides
persistent coding rules so you don't have to re-explain the codebase each
time. Keep this file in sync with `CLAUDE.md`.

**Map:** [`README.md`](README.md) (install, CLI, tech stack).
**Live pipeline:** [`docs/architecture.md`](docs/architecture.md).
**Original spec:** [`docs/strategy.md`](docs/strategy.md).
**Tree:** [`docs/file-structure.md`](docs/file-structure.md).
**Data:** [`docs/data.md`](docs/data.md).
**Results:** [`docs/diagnostics.md`](docs/diagnostics.md).
**Knobs:** `src/config.py` — import `CONFIG`; do not copy defaults into new code.

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
Every function needs a one-line summary, Args, and Returns.

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
signatures without updating this file and `docs/architecture.md`.

```python
# src/data/load.py
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

load_earnings_dates() -> pd.DataFrame
    # columns: [ticker, earnings_date] (tz-naive normalized Timestamps)
    # empty frame (with columns) when data/raw/earnings.parquet is missing —
    # callers must treat that as "earnings features disabled"

# src/clustering/correlation.py
build_distance_matrix(returns: pd.DataFrame, window: int) -> pd.DataFrame
    # NxN DataFrame, index and columns are ticker strings
    # values are (1 - correlation), range [0, 2]

# src/clustering/kmeans.py
run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]
    # keys: cluster id (int)
    # values: list of ticker strings in that cluster

# src/scoring/candidate_pairs.py
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
    # CONFIG.finalists_per_cluster pairs per cluster, only pairs scoring >= CONFIG.min_composite_score
    # after half-life hard gate, min_cointegration_score, and β_F > min_formation_beta
    # raw absolute scores — no normalization
    # ticker_a is canonical stock A based on halflife direction
    # empty result must keep these columns; engine must not clear active_pairs

# src/signals/entry_exit.py
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
    # "STOP_LOSS", "PLATEAU_STOP", "TIME_STOP", "HOLD"
    # z uses locked formation β/μ/σ only
    # entries require |z| within [entry_zscore, entry_zscore_max]
    # PLATEAU_STOP: adverse z >= stop_plateau_zscore for stop_plateau_days
    # consecutive days (engine applies the STOP_LOSS cooldown to it)

# src/backtest/engine.py
run_backtest(config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    # (trade_log, nav_series, pair_daily_mtm, blocked_entries)
    # blocked_entries: [date, ticker_a, ticker_b, signal, reason]
    # trade_log exit actions include PLATEAU_STOP, EARNINGS_EXIT, and DOLLAR_STOP
```

---

## Key Parameters

Import from `src/config.py`. Snapshot of live defaults (re-read the file if
unsure):

| Parameter | Default | Notes |
|---|---|---|
| `clustering_window` | 120 days | Correlation matrix for K-means |
| `formation_window` | 120 days | Locked β / μ / σ for z-score |
| `signal_window` | 60 days | Live hedge β for share resize only — not for z |
| `johansen_window` | 252 days | Johansen lookback |
| `k_min` / `k_max` | 4 / 6 | Silhouette-scored k |
| `entry_zscore` | 1.25 | Absolute formation z to enter |
| `entry_requires_cross` | True | Enter only the day z first crosses the band |
| `entry_zscore_max` | 2.0 | Entry band upper cap; None disables |
| `take_profit_zscore` | 1.0 | Absolute z to take profit |
| `stop_loss_zscore` | 3.5 | Absolute z to stop |
| `stop_plateau_zscore` / `stop_plateau_days` | 2.75 / 3 | Consecutive adverse-z exit; days=0 disables |
| `earnings_exit_days_before` | 2 | Close a losing pair before a leg reports; 0 disables |
| `earnings_exit_min_adverse_z` | 1.75 | Adverse-z floor for the pre-earnings exit |
| `time_stop_days` | 50 | Force close |
| `max_pair_loss_pct` | None | Optional per-pair dollar loss cap; exit `DOLLAR_STOP` |
| `momentum_window` / `momentum_threshold` | 14 / 0.15 | Block entries if either leg moved this much |
| `pair_stop_cooldown_days` | 20 | Trading days after STOP_LOSS |
| `johansen_threshold` | 0.10 | Mapping aid; not a BH kill switch |
| `min_cointegration_score` | 0.70 | Soft floor on `1 − p_adj` |
| `min_formation_beta` | 0.0 | Drop non-positive formation β |
| `halflife_min` / `halflife_max` | 5 / 20 | Hard gate |
| `vix_entry_block` / `vix_resume` | 28 / 25 | Resume must hold 5 days |
| `weight_correlation_stability` | 0.20 | |
| `weight_cointegration` | 0.25 | |
| `weight_halflife` | 0.30 | Binary gate + score |
| `weight_volatility` | 0.15 | |
| `weight_fundamentals` | 0.10 | Sector labels, not P/S |
| `min_composite_score` | 0.55 | Absolute floor after raw weighted sum |
| `finalists_per_cluster` | 2 | Pairs per cluster; also caps concurrent positions |
| `rebalance_beta_intra_trade` | False | Intra-trade B-leg resize disabled |
| `beta_rebalance_threshold` | 0.15 | Only if rebalance enabled; never overwrite `beta_at_entry` |
| `max_weight_per_pair` | 0.35 | Per-leg allocation cap as fraction of NAV |
| `target_concurrent_pairs` | 10 | Sizing divisor cap |
| `min_dollar_volume` | 25,000,000 | 30-day ADV$ proxy |
| `backtest_start_date` / `backtest_end_date` | 2022-01-01 / 2024-12-31 | Simulation window only |
| `initial_capital` | 100000 | |
| `random_seed` | 42 | K-means |

Install, CLI, and tests: [`README.md`](README.md). Git style:
[`docs/git-conventions.md`](docs/git-conventions.md).

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

**Script 04 is the report** — It charts script 03 CSVs. There is no
walk-forward slice step.

**Percentile thresholds are gone** — Signals are z-score. Do not
reintroduce `entry_percentile_*` in new code unless config and tests
change together.

**Drawdown halt is a cooldown, not a permanent stop** — After the trim
completes, `Portfolio.check_drawdown_controls` resets `peak_nav` to current
NAV and releases the halt after `drawdown_recovery_days` non-losing days.
Do not reintroduce the old "recover to within 5% of the all-time peak"
release rule.

**Trim is per-day multiplicative** — `_execute_trim` applies
`drawdown_trim_factor ** (1/drawdown_trim_days)` per day so the position
reaches the target after the trim window. Applying the full factor daily
compounds to near-liquidation.

**Sizing levers are data-tested** — Concentrating capital
(`target_concurrent_pairs` 4) and per-pair dollar caps (`max_pair_loss_pct`
0.02–0.05) both lowered NAV and win rate in the Round 3 test matrix.
Re-test before changing either; see `docs/diagnostics.md`.

**Plateau stop is a persistence rule, not a tighter stop** — Hard stops at
2.0–3.0 test worse because single-day spikes usually revert. `PLATEAU_STOP`
fires only after `stop_plateau_days` consecutive adverse days at
`≥ stop_plateau_zscore`. Do not "simplify" it into a lower `stop_loss_zscore`.

**Earnings features need data/raw/earnings.parquet** — fetched via
`scripts/01_fetch_data.py --stage earnings` (needs `lxml`). When the file
is missing the engine logs a warning and silently disables the pre-earnings
exit. Only *losing* positions are exited before a print
(`earnings_exit_min_adverse_z`).

**Scoring reads the module-global CONFIG** — `composite.py` uses
`CONFIG.min_cointegration_score` / `finalists_per_cluster` from its own
import, not the config passed to `run_backtest`. Sensitivity scripts that
vary scoring parameters must patch `src.scoring.composite.CONFIG`.

---

## Questions and Decisions

Before making a non-obvious design decision, check `docs/open-questions.md`.
After deciding, log it in `docs/decisions.md`.

If Codex makes a mistake — wrong window, wrong column name, wrong interface
— add a correction to this file under a new "Corrections" section so it
does not repeat the error in future sessions.
