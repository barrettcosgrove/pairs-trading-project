# ARQ Pairs Trading — Module Implementation Checklist

> **Historical build checklist.** Several items still describe percentiles,
> `walkforward_results.csv`, and `k_max = 20`. Live behavior is
> `src/config.py` + [`architecture.md`](architecture.md). Do not implement
> new work from unchecked boxes without checking the code first.

This document tells each teammate exactly what to build, in what order,
and what each file must do. Work top to bottom within your section.
Check off each item as you complete it.

Read `CLAUDE.md` before starting any file. Start every Claude Code session
by telling it to read `CLAUDE.md` first.

---

## Barrett — Data Layer (COMPLETE)

All files below are done and merged. No action needed.

- [x] `src/config.py` — all strategy parameters as frozen dataclass
- [x] `src/data/fetch.py` — yfinance batch downloader
- [x] `src/data/clean.py` — price validation and log return computation
- [x] `src/data/load.py` — typed data access layer
- [x] `src/universe/filter.py` — five hard pre-filters, monthly reconstitution
- [x] `data/sector_map.py` — ticker → subsector mapping
- [x] `scripts/01_fetch_data.py` — fetch orchestrator
- [x] `scripts/02_build_universe.py` — build orchestrator

**Before you start:** Pull latest, run `uv sync`, download `data/raw/`
from Google Drive, run `scripts/02_build_universe.py`, confirm
`sanity_check.py` shows 11 passed.

---

## Althan — Clustering and Scoring

Build in this exact order. Each file depends on the previous.

### Step 1: `src/clustering/correlation.py`

**What it does:** Takes daily log returns and builds a pairwise distance
matrix used as input to K-means clustering.

Checklist:
- [x] Function: `build_distance_matrix(returns: pd.DataFrame, window: int) -> pd.DataFrame`
- [x] Computes Pearson correlation matrix from log returns over trailing `window` days
- [x] Converts to distance matrix: `D[i,j] = 1 - correlation[i,j]`
- [x] Returns NxN DataFrame where index and columns are ticker strings
- [x] Values range from 0 (perfectly correlated) to 2 (perfectly anti-correlated)
- [x] Uses `CONFIG.clustering_window` (120 days) — not `CONFIG.signal_window`
- [x] Saves monthly snapshots to `data/processed/correlation_matrices/YYYY-MM.parquet`
- [x] Docstring on every function

Test: Matrix is symmetric, diagonal is all zeros, values in [0, 2].

---

### Step 2: `src/clustering/kmeans.py`

**What it does:** Runs K-means on the distance matrix, selects the best k
using silhouette scoring, returns cluster assignments.

Checklist:
- [x] Function: `run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]`
- [x] Scans k from `CONFIG.k_min` (4) to `CONFIG.k_max` (currently 6)
- [x] Runs `CONFIG.kmeans_restarts` (10) random restarts per k
- [x] Selects k with highest average silhouette score
- [x] Uses `CONFIG.random_seed` (42) for reproducibility
- [x] Returns dict: `{cluster_id: [ticker1, ticker2, ...]}`
- [x] Logs winning k and silhouette score at INFO level
- [x] Validates distance matrix is square, symmetric, zero-diagonal, and in [0, 2]
- [x] Docstring on every function

Test: Every input ticker appears in exactly one cluster. No empty clusters.

---

### Step 3: `src/scoring/candidate_pairs.py`

**What it does:** Expands cluster assignments into all unique within-cluster
candidate pairs for downstream scoring.

Checklist:
- [x] Function: `build_candidate_pairs(clusters: dict[int, list[str]]) -> pd.DataFrame`
- [x] Returns columns: `[ticker_a, ticker_b, cluster_id]`
- [x] Generates unique unordered within-cluster pairs only
- [x] Skips clusters with fewer than 2 tickers
- [x] Validates that each ticker appears in exactly one cluster
- [x] Docstring on every function

Test: Cluster of n tickers returns n choose 2 rows. Singleton clusters contribute zero rows.

---

### Step 4: `src/scoring/correlation_stability.py`

**What it does:** Scores how stable each pair's correlation has been recently
vs historically.

Checklist:
- [x] Function: `score(ticker_a: str, ticker_b: str, returns: pd.DataFrame, as_of: date) -> float`
- [x] Uses only return data available on or before `as_of`
- [x] Computes recent correlation over `CONFIG.signal_window` (60 days)
- [x] Computes historical correlation over `CONFIG.correlation_stability_historical_window` (252 days)
- [x] Returns score in [0, 1]
- [x] Returns 0.0 if recent correlation < `CONFIG.min_recent_correlation` (0.50)
- [x] Uses bounded stability score: `1 - abs(recent_corr - historical_corr)`
- [x] Optional batch helper: `score_candidate_pairs(candidate_pairs, returns, as_of) -> pd.DataFrame`
- [x] No hardcoded numbers — use CONFIG
- [x] Docstring with Args and Returns

---

### Step 5: `src/scoring/halflife.py`

**What it does:** Estimates how many days it takes the spread to revert
halfway to its mean using AR(1) regression.

Checklist:
- [ ] Function: `score(ticker_a: str, ticker_b: str, returns: pd.DataFrame, as_of: date) -> float`
- [ ] Fits AR(1) regression: `Δspread_t = α + β × spread_{t-1} + ε`
- [ ] Computes half-life: `log(2) / abs(β)`
- [ ] Returns 1.0 if half-life in [`CONFIG.halflife_min`, `CONFIG.halflife_max`] (5–20 days)
- [ ] Returns 0.0 if slope is positive (diverging pair)
- [ ] Returns 0.0 if half-life outside acceptable range
- [ ] Docstring with Args and Returns

---

### Step 6: `src/scoring/cointegration.py`

**What it does:** Tests whether the spread between two stocks is stationary
using the Johansen test. Core of the composite score at 30% weight.

Checklist:
- [ ] Function: `score(ticker_a: str, ticker_b: str, prices: pd.DataFrame, as_of: date) -> float`
- [ ] Runs Johansen trace test on log prices of both stocks
- [ ] Applies Benjamini-Hochberg FDR correction across all pairs in the cluster
- [ ] Returns score in [0, 1] — lower p-value scores higher
- [ ] Returns 0.0 if BH-corrected p-value >= `CONFIG.johansen_threshold` (0.10)
- [ ] Uses `statsmodels.tsa.vector_ar.vecm.coint_johansen`
- [ ] Docstring with Args and Returns

---

### Step 7: `src/scoring/volatility.py`

**What it does:** Checks whether two stocks have compatible volatility levels
using a dual-window approach.

Checklist:
- [ ] Function: `score(ticker_a: str, ticker_b: str, returns: pd.DataFrame, as_of: date) -> float`
- [ ] Computes volatility ratio (larger/smaller) over 20-day and 120-day windows
- [ ] Combines: 60% short-window + 40% long-window (`CONFIG.volatility_short_weight`)
- [ ] Returns 0.0 if short-window ratio > `CONFIG.max_volatility_ratio` (2.5)
- [ ] Lower combined ratio scores higher
- [ ] Docstring with Args and Returns

---

### Step 8: `src/scoring/fundamentals.py`

**What it does:** Checks whether two companies are valued similarly using
P/S ratio and revenue growth.

Checklist:
- [ ] Function: `score(ticker_a: str, ticker_b: str, fundamentals: pd.DataFrame) -> float`
- [ ] Reads from `load_fundamentals()` — do not call `pd.read_parquet()` directly
- [ ] Computes P/S ratio similarity: higher/lower ratio, target near 1.0
- [ ] Computes revenue growth difference: absolute difference in growth rates
- [ ] Returns neutral score 0.5 if either ticker has NaN fundamentals
- [ ] Applies mild penalty when both stocks exceed `CONFIG.ps_ratio_high_threshold` (30x)
- [ ] Docstring with Args and Returns

---

### Step 9: `src/scoring/composite.py`

**What it does:** Aggregates all five component scores, applies a minimum score
threshold, and returns exactly 1 finalist per cluster.

Checklist:
- [ ] Function: `score_candidates(clusters: dict, returns: pd.DataFrame, prices: pd.DataFrame, as_of: date) -> pd.DataFrame`
- [ ] Imports and calls all five component scorers
- [ ] Applies global pre-filter gates before scoring (see CLAUDE.md data contracts)
- [ ] Computes an absolute weighted score from raw [0,1] component scores
- [ ] Combines with weights from CONFIG: 0.20 / 0.30 / 0.25 / 0.15 / 0.10
- [ ] Returns the top 1 pair per cluster, ONLY IF its score > `CONFIG.min_composite_score`
- [ ] Output columns: `[ticker_a, ticker_b, cluster_id, composite_score]`
- [ ] Sorted by composite_score descending
- [ ] Docstring with Args and Returns

Test: Weights sum to 1.0. All scores in [0,1]. No cross-cluster pairs in output.

---

## Anvay — Signals, Regime, Backtest

Build in this exact order.

### Step 1: `src/signals/hedge_ratio.py`

**What it does:** Estimates the hedge ratio β via rolling OLS regression
of log prices.

Checklist:
- [ ] Function: `compute(ticker_a: str, ticker_b: str, prices: pd.DataFrame, as_of: date) -> tuple[float, float]`
- [ ] Fits OLS: `log(P_A) = α + β × log(P_B) + ε` over `CONFIG.signal_window` (60) days
- [ ] Returns (β, α) tuple
- [ ] Flags if β has flipped sign since entry
- [ ] Sets rebalance flag if `|β_today - β_entry| > CONFIG.beta_rebalance_threshold` (0.15)
- [ ] Uses log prices — not raw prices
- [ ] Docstring with Args and Returns

---

### Step 2: `src/signals/spread.py`

**What it does:** Computes the spread between two stocks and its position
in the empirical percentile distribution.

Checklist:
- [ ] Function: `compute(ticker_a: str, ticker_b: str, beta: float, prices: pd.DataFrame, as_of: date) -> tuple[float, float]`
- [ ] Computes: `spread = log(P_A) - β × log(P_B)`
- [ ] Maintains rolling 60-day empirical distribution of spread
- [ ] Returns (spread_value, percentile) tuple
- [ ] All three parameters (β, μ, σ) must use same `CONFIG.signal_window` window
- [ ] Uses log prices throughout
- [ ] Docstring with Args and Returns

---

### Step 3: `src/signals/entry_exit.py`

**What it does:** Generates trading signals based on where the spread sits
in its empirical percentile distribution.

Checklist:
- [ ] Function: `get_signal(ticker_a: str, ticker_b: str, prices: pd.DataFrame, as_of: date, days_open: int) -> str`
- [ ] Entry long: spread < `CONFIG.entry_percentile_low` (2nd percentile)
- [ ] Entry short: spread > `CONFIG.entry_percentile_high` (98th percentile)
- [ ] Take profit: spread re-enters 40th–60th percentile
- [ ] Stop loss: spread > 99.5th or < 0.5th percentile
- [ ] Time stop: `days_open > CONFIG.time_stop_days` (20)
- [ ] Returns one of: `LONG_SPREAD`, `SHORT_SPREAD`, `TAKE_PROFIT`, `STOP_LOSS`, `TIME_STOP`, `HOLD`
- [ ] Docstring with Args and Returns

---

### Step 4: `src/regime/vix.py`

**What it does:** Portfolio-level filter — blocks all new entries when
market volatility is elevated.

Checklist:
- [ ] Function: `new_entries_permitted(as_of: date, vix_series: pd.Series) -> bool`
- [ ] Blocks entries when VIX > `CONFIG.vix_entry_block` (28.0)
- [ ] Resumes when VIX stays below `CONFIG.vix_resume` (25.0) for `CONFIG.vix_resume_days` (5) days
- [ ] Reads VIX via `load_vix()` from `src/data/load.py`
- [ ] Docstring with Args and Returns

---

### Step 5: `src/regime/earnings.py`

**What it does:** Blocks new entries around earnings periods.

Checklist:
- [ ] Function: `in_blackout(ticker: str, as_of: date) -> bool`
- [ ] Blocks entries during last 5 trading days of March, June, September, December
- [ ] Blocks for 1 trading day after quarter end
- [ ] Returns True if either ticker in a pair is in blackout
- [ ] Documented as end-of-quarter approximation (not per-company dates)
- [ ] Docstring with Args and Returns

---

### Step 6: `src/backtest/portfolio.py`

**What it does:** Tracks all open positions, capital, and NAV.

Checklist:
- [ ] Class or functions to open/close positions
- [ ] Sizes positions equally across all active pairs (NAV / max_active_pairs)
- [ ] Enforces dollar-neutral sizing: long leg = $V, short leg = $V × β
- [ ] Enforces no-shared-stocks constraint
- [ ] Implements drawdown controls (>5% → 50% size reduction, >10% → halt)
- [ ] Computes NAV daily from positions + cash
- [ ] Docstring on every function

---

### Step 7: `src/backtest/costs.py`

**What it does:** Applies transaction costs to every trade.

Checklist:
- [ ] Function: `apply_costs(trade: dict) -> float`
- [ ] Commission: `CONFIG.commission_per_share` ($0.005/share)
- [ ] Slippage: 5 bps (ADV > 5M) or 10 bps (1–5M ADV)
- [ ] Bid-ask: 2 bps (price > $30) or 5 bps ($10–30)
- [ ] Short borrow: `CONFIG.short_borrow_annual` (2% annualized) on short legs
- [ ] Skips trade if expected profit < 2× round-trip cost — logs the skip
- [ ] Docstring with Args and Returns

---

### Step 8: `src/backtest/execution.py`

**What it does:** Simulates order fills.

Checklist:
- [ ] Function: `execute(order: dict, prices: pd.DataFrame, as_of: date) -> dict`
- [ ] Uses market-on-close prices
- [ ] Short leg submitted first
- [ ] Cancels long leg if short fills < 95% — queues retry next day
- [ ] Returns fill prices and quantities
- [ ] Docstring with Args and Returns

---

### Step 9: `src/backtest/engine.py`

**What it does:** The main simulation loop — iterates day by day, applies
all strategy logic, and produces the trade log and NAV series.

Checklist:
- [ ] Daily loop from backtest start to end date
- [ ] Monthly: reconstitute universe
- [ ] Quarterly: re-cluster and re-score
- [ ] Daily: update β/spread/percentiles, check regime filters, generate signals
- [ ] Calls `entry_exit.py` for signals, `portfolio.py` for positions, `costs.py` for costs
- [ ] Logs all decisions including signals blocked by filters
- [ ] Writes `outputs/backtest_results/trade_log.csv`
- [ ] Writes `outputs/backtest_results/nav_series.csv`
- [ ] No look-ahead bias — only uses data available on each day

---

## Nanshu — Metrics and Reporting

### Step 1: `src/metrics/performance.py`

Checklist:
- [ ] Function: `compute(trade_log: pd.DataFrame, nav_series: pd.DataFrame) -> dict`
- [ ] Annualized return net of costs
- [ ] Sharpe ratio (target > 1.50)
- [ ] Sortino ratio (target > 2.0)
- [ ] Maximum drawdown (target < 10%)
- [ ] Win rate (target 60–70%)
- [ ] Average trade duration vs half-life
- [ ] Exit type breakdown (take profit / stop loss / time stop / cointegration break)
- [ ] Average active pairs per month
- [ ] Docstring with Args and Returns

---

### Step 2: `src/metrics/reporting.py`

Checklist:
- [ ] NAV curve vs XLK benchmark
- [ ] Monthly returns heatmap
- [ ] Drawdown chart
- [ ] Pairs activity timeline
- [ ] Exit type breakdown pie chart
- [ ] Walk-forward results table
- [ ] All charts saved to `outputs/report/` as PNG
- [ ] Docstring on every function

---

### Step 3: `scripts/03_run_backtest.py`

Checklist:
- [ ] Calls `src/backtest/engine.py`
- [ ] Accepts `--config` flag for sensitivity analysis
- [ ] Prints summary: Sharpe, drawdown, win rate, number of trades
- [ ] Expected runtime < 15 minutes

---

### Step 4: `scripts/04_walkforward.py`

Checklist:
- [ ] Divides training period into quarterly windows
- [ ] Re-selects pairs on each rolling training set
- [ ] Evaluates on following quarter
- [ ] Writes `outputs/backtest_results/walkforward_results.csv`

---

### Step 5: `scripts/05_generate_report.py`

Checklist:
- [ ] Reads backtest and walkforward results
- [ ] Calls `src/metrics/reporting.py` to generate all charts
- [ ] Writes everything to `outputs/report/`
