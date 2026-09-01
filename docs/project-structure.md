# ARQ Pairs Trading — Project File Structure

How to read this document: every path has a role and a short description of
what it does **in the current code**. Live parameters are in `src/config.py`.
Empirical backtest notes are in `docs/diagnostics.md`.

The tree itself lives in [`docs/file-structure.md`](file-structure.md).

---

## Root

| File | Role | Description |
|---|---|---|
| `README.md` | Entry point | Install, fetch, run 01→04. |
| `CLAUDE.md` / `AGENTS.md` | AI context | Conventions, contracts, ownership. Keep in sync with code. |
| `pyproject.toml` / `uv.lock` | Dependencies | Managed with `uv`. |
| `.gitignore` | VCS filter | Ignores `data/raw/`, `data/processed/`, `outputs/`, venvs. |
| `.env.example` | Env template | Copy to `.env`. Used for `ALPHA_VANTAGE_KEY` (optional SPY fetch). |
| `sanity_check.py` | Local helper | Ad-hoc environment check. |
| `description.md` | Notes | Informal project description. |
| `file-structure.md` | Pointer | Redirects to `docs/file-structure.md`. |

---

## `docs/`

| File | Role |
|---|---|
| `strategy.md` | Original v2.0 product spec. Several sections (percentiles, tiering, Bollinger) are **not** what the engine does. Treat as design history. |
| `architecture.md` | Implemented pipeline, data flow, contracts. |
| `data.md` | Sources, parquet schemas, known data issues. |
| `diagnostics.md` | Backtest issues, attempted fixes, performance by run. |
| `decisions.md` | Decision log. |
| `open-questions.md` | Unresolved items. |
| `implementation-plan.md` | Original two-week build plan. |
| `file-structure.md` | Canonical directory tree. |
| `project-structure.md` | This inventory. |
| `git-conventions.md` | Branch / commit / PR conventions. |
| `llm_guide.md` | Team LLM usage notes. |
| `module_checlist.md` | Implementation checklist (may lag the code). |

---

## `data/`

| Path | Role |
|---|---|
| `data/sector_map.py` | Committed ticker → sector labels. Used by universe filter and `src/scoring/fundamentals.py`. |
| `data/raw/prices.parquet` | Daily OHLCV for `CANDIDATE_TICKERS` (~95 names). Fetched 2019-01-01 → 2026-04-01. |
| `data/raw/fundamentals.parquet` | Current P/S and TTM growth snapshot. **Not used by live scoring.** |
| `data/raw/regime.parquet` | Daily VIX + SPY. VIX from CBOE (yfinance fallback); SPY from Alpha Vantage if keyed, else yfinance, else synthetic. |
| `data/processed/returns.parquet` | Log returns from `clean.py`. |
| `data/processed/universe_history.parquet` | Point-in-time universe membership by reconstitution date. |
| `data/processed/correlation_matrices/` | Optional monthly distance-matrix cache from `correlation.py`. |

`load.py` is allowed to read these parquets. No other production module should call `pd.read_parquet()`.

---

## `src/config.py`

Frozen `StrategyConfig`. Every tunable lives here. Import `CONFIG`.

Notable live defaults (not the original percentile spec):

| Area | Defaults |
|---|---|
| Universe | price > $10, ADV > 1M, dollar volume > $25M, SPY ρ < 0.90 |
| Clustering | 120-day window, k ∈ [4, 6], refresh 63 calendar days |
| Scoring weights | corr 0.20 / coint 0.25 / half-life 0.30 / vol 0.15 / sector 0.10 |
| Gates | half-life 5–20 days; `min_cointegration_score` 0.40; `min_composite_score` 0.55; formation β > 0 |
| Signals | formation z: enter 1.5, take-profit 0.5, stop 3.5; time stop 50 days; momentum 14d / 15% |
| Regime | VIX block 28 / resume 25 for 5 days; quarter-end earnings blackout |
| Backtest | $100k, 2022-01-01 → 2024-12-31, `oos_fraction` 0.30 |

---

## `src/data/`

| File | Description |
|---|---|
| `fetch.py` | Only file that imports `yfinance` / `curl_cffi`. Batch prices, per-ticker fundamentals, VIX/SPY with fallbacks. |
| `clean.py` | Quality gates, writes returns + `outputs/data_quality_report.txt`. |
| `load.py` | `load_prices`, `load_returns`, `load_vix`, `load_universe`, `load_fundamentals`. `start`/`end` may be `None`. |

---

## `src/universe/`

`filter.py` — monthly hard filters: in `sector_map`, ADV, price, dollar volume, SPY correlation. Soft check: ≥8 subsectors. Writes `universe_history.parquet`.

---

## `src/clustering/`

| File | Description |
|---|---|
| `correlation.py` | 120-day Pearson distance `D = 1 − ρ`. |
| `kmeans.py` | Silhouette-scored K-means, k ∈ [`k_min`, `k_max`]. |

Only within-cluster pairs go to scoring.

---

## `src/scoring/`

| File | Description |
|---|---|
| `candidate_pairs.py` | All unordered within-cluster pairs. |
| `correlation_stability.py` | Recent vs 252-day correlation. 0 if recent ρ < 0.50. |
| `cointegration.py` | Bidirectional Johansen on `johansen_window` (252) days, BH within cluster, score = `1 − p_adj`. Not a BH p < 0.10 kill switch. |
| `halflife.py` | AR(1) half-life; 1.0 inside [5, 20], else 0. Hard gate. |
| `volatility.py` | Dual-window vol ratio; discard if short-window ratio > 2.5. |
| `fundamentals.py` | Sector match: 1.0 same / 0.4 cross / 0.5 unknown. No P/S. |
| `composite.py` | Weighted sum, half-life + min coint floor, min composite 0.55, drop β ≤ 0, top 1 pair per cluster. Adds formation β/μ/σ and half-life value. |

---

## `src/signals/`

| File | Description |
|---|---|
| `hedge_ratio.py` | Live 60-day OLS β (used for share rebalance, not for z). |
| `spread.py` | `s = log A − β_F log B`, formation z vs locked μ_F, σ_F. |
| `entry_exit.py` | z-score signals + 14-day momentum block on entries. |

---

## `src/regime/`

| File | Description |
|---|---|
| `vix.py` | Block new entries when VIX > 28; resume after 5 days < 25. |
| `earnings.py` | Quarter-end blackout (not real earnings calendars). |

---

## `src/backtest/`

| File | Description |
|---|---|
| `engine.py` | Daily loop. Loads warmup history; trades only [`backtest_start_date`, `backtest_end_date`]. Monthly universe, quarterly cluster/score. Keeps last pairs if scoring is empty. z uses locked formation β/μ/σ; `beta_hedge` for share resize. |
| `portfolio.py` | Positions, NAV, drawdown controls, `beta_hedge`. |
| `costs.py` | Commission, slippage, bid-ask, borrow; profit-to-cost skip. |
| `execution.py` | Simulated fills, next-day retry. |

---

## `src/metrics/`

Stubs. Do not expect charts from script 05 yet.

---

## Leftover / non-pipeline code

| Path | Description |
|---|---|
| `src/tiering/` | Old confirmation helper. Engine does not call it. |
| `src/scrap/` | Copied prototype backtester / price cache. |
| `working_model/` | Earlier standalone prototype. |
| `scratch/` | One-off NAV / OOS / plot scripts. |

---

## `scripts/`

| Script | What it actually does |
|---|---|
| `01_fetch_data.py` | CLI around `fetch_all`. Stages: prices → regime → fundamentals. |
| `02_build_universe.py` | Clean + monthly universe history. |
| `03_run_backtest.py` | `run_backtest(CONFIG)` over the full configured calendar. Writes `trade_log.csv`, `nav_series.csv`, `pair_daily_mtm.csv`. |
| `04_walkforward.py` | Slices the last `oos_fraction` of those CSVs. Does **not** re-run rolling quarterly re-fits. |
| `05_generate_report.py` | Stub. |

---

## `tests/`

Synthetic only. No yfinance, no `data/` parquet.

`test_backtest.py`, `test_clustering.py`, `test_cointegration.py`, `test_composite.py`, `test_correlation_stability.py`, `test_halflife.py`, `test_scoring.py`, `test_scoring_integration.py`, `test_signals.py`, `test_volatility.py`.

---

## `outputs/` (gitignored)

| File | Writer |
|---|---|
| `backtest_results/trade_log.csv` | Script 03 |
| `backtest_results/nav_series.csv` | Script 03 |
| `backtest_results/pair_daily_mtm.csv` | Script 03 |
| `backtest_results/oos_*.csv` | Script 04 |
| `data_quality_report.txt` | Script 02 / `clean.py` |
| `report/` | Intended for script 05 (empty) |
