# File Structure

Canonical tree of the repository and a short note on what each path does.
Generated data, virtual environments, caches, and report outputs are
gitignored and regenerated locally.

Live parameters: [`src/config.py`](../src/config.py).
Implemented pipeline: [`architecture.md`](architecture.md).
Original spec: [`strategy.md`](strategy.md).
Data schemas: [`data.md`](data.md).
Backtest rounds: [`diagnostics.md`](diagnostics.md).
Install and CLI: [`README.md`](../README.md).

```text
pairs-trading/
├── .env.example
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── sanity_check.py
├── uv.lock
│
├── data/
│   ├── sector_map.py              # committed ticker → sector labels
│   ├── raw/                       # gitignored
│   │   ├── prices.parquet
│   │   ├── fundamentals.parquet
│   │   ├── regime.parquet
│   │   ├── earnings.parquet
│   │   ├── fetch_manifest.json
│   │   └── fundamentals_checkpoint.parquet
│   └── processed/                 # gitignored
│       ├── returns.parquet
│       ├── universe_history.parquet
│       └── correlation_matrices/  # optional monthly D = 1−ρ cache
│
├── docs/
│   ├── architecture.md            # live pipeline and contracts
│   ├── data.md                    # sources and parquet schemas
│   ├── decisions.md               # decision log
│   ├── diagnostics.md             # issues and backtest rounds
│   ├── file-structure.md          # this file
│   ├── git-conventions.md
│   ├── open-questions.md
│   └── strategy.md                # original v2.0 spec (not all live)
│
├── outputs/                       # gitignored
│   ├── backtest_results/
│   │   ├── trade_log.csv
│   │   ├── nav_series.csv
│   │   ├── pair_daily_mtm.csv
│   │   └── blocked_entries.csv
│   ├── report/                    # script 04 charts + metrics_summary.txt
│   └── data_quality_report.txt
│
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_build_universe.py
│   ├── 03_run_backtest.py
│   └── 04_generate_report.py
│
├── src/
│   ├── config.py
│   ├── backtest/
│   │   ├── costs.py
│   │   ├── engine.py
│   │   ├── execution.py
│   │   └── portfolio.py
│   ├── clustering/
│   │   ├── correlation.py
│   │   └── kmeans.py
│   ├── data/
│   │   ├── clean.py
│   │   ├── fetch.py
│   │   └── load.py
│   ├── metrics/
│   │   ├── performance.py
│   │   └── reporting.py
│   ├── regime/
│   │   ├── earnings.py
│   │   └── vix.py
│   ├── scoring/
│   │   ├── candidate_pairs.py
│   │   ├── cointegration.py
│   │   ├── composite.py
│   │   ├── correlation_stability.py
│   │   ├── fundamentals.py
│   │   ├── halflife.py
│   │   └── volatility.py
│   ├── signals/
│   │   ├── entry_exit.py
│   │   ├── hedge_ratio.py
│   │   └── spread.py
│   ├── tiering/                   # leftover; not called by the engine
│   ├── scrap/                     # old prototypes; not the live pipeline
│   └── universe/
│       └── filter.py
│
├── tests/                         # synthetic pytest only
├── scratch/                       # ad-hoc analysis
└── working_model/                 # earlier prototype; not wired to scripts/
```

---

## Root

| File | Description |
|---|---|
| `README.md` | High-level map, install, CLI. |
| `CLAUDE.md` / `AGENTS.md` | AI assistant coding rules and contracts. Keep in sync with the code. |
| `pyproject.toml` / `uv.lock` | Dependencies, managed with `uv`. |
| `.gitignore` | Ignores `data/raw/`, `data/processed/`, `outputs/`, venvs. |
| `.env.example` | Copy to `.env`. Optional `ALPHA_VANTAGE_KEY` for SPY fetch. |
| `sanity_check.py` | Ad-hoc environment check. |

## `docs/`

| File | Description |
|---|---|
| `strategy.md` | Original v2.0 product spec. Percentiles, tiering, and Bollinger are **not** what the engine does. |
| `architecture.md` | Implemented pipeline, data flow, contracts. |
| `data.md` | Sources, parquet schemas, known data issues. |
| `diagnostics.md` | Backtest issues, attempted fixes, performance by run. |
| `decisions.md` | Decision log. |
| `open-questions.md` | Unresolved items. |
| `file-structure.md` | This inventory. |
| `git-conventions.md` | Branch / commit / PR conventions. |

## `data/`

| Path | Description |
|---|---|
| `data/sector_map.py` | Committed ticker → sector labels. Used by the universe filter and `src/scoring/fundamentals.py`. |
| `data/raw/prices.parquet` | Daily OHLCV for `CANDIDATE_TICKERS`. |
| `data/raw/fundamentals.parquet` | Current P/S and TTM growth snapshot. **Not used by live scoring.** |
| `data/raw/regime.parquet` | Daily VIX + SPY. |
| `data/raw/earnings.parquet` | Per-ticker earnings dates (optional; pre-earnings exit disables if missing). |
| `data/processed/returns.parquet` | Log returns from `clean.py`. |
| `data/processed/universe_history.parquet` | Point-in-time universe membership by reconstitution date. |
| `data/processed/correlation_matrices/` | Optional monthly distance-matrix cache from `correlation.py`. |

`load.py` is the only production module that should call `pd.read_parquet()`.
Schemas: [`data.md`](data.md).

## `src/config.py`

Frozen `StrategyConfig`. Every tunable lives here. Import `CONFIG`.

## `src/data/`

| File | Description |
|---|---|
| `fetch.py` | Only file that imports `yfinance` / `curl_cffi`. Batch prices, per-ticker fundamentals, VIX/SPY with fallbacks. |
| `clean.py` | Quality gates; writes returns and `outputs/data_quality_report.txt`. |
| `load.py` | `load_prices`, `load_returns`, `load_vix`, `load_universe`, `load_fundamentals`, `load_earnings_dates`. `start`/`end` may be `None`. |

## `src/universe/`

`filter.py` — monthly hard filters: in `sector_map`, ADV, price, dollar volume, SPY correlation. Writes `universe_history.parquet`.

## `src/clustering/`

| File | Description |
|---|---|
| `correlation.py` | Pearson distance `D = 1 − ρ`. |
| `kmeans.py` | Silhouette-scored K-means, k ∈ [`k_min`, `k_max`]. |

Only within-cluster pairs go to scoring.

## `src/scoring/`

| File | Description |
|---|---|
| `candidate_pairs.py` | All unordered within-cluster pairs. |
| `correlation_stability.py` | Recent vs 252-day correlation. |
| `cointegration.py` | Bidirectional Johansen, BH within cluster, score = `1 − p_adj`. |
| `halflife.py` | AR(1) half-life; hard gate outside [`halflife_min`, `halflife_max`]. |
| `volatility.py` | Dual-window vol ratio. |
| `fundamentals.py` | Sector match scores. No P/S. |
| `composite.py` | Weighted sum, gates, formation β/μ/σ, finalists per cluster. |

## `src/signals/`

| File | Description |
|---|---|
| `hedge_ratio.py` | Live OLS β (share rebalance only, not z). |
| `spread.py` | `s = log A − β_F log B`, formation z vs locked μ_F, σ_F. |
| `entry_exit.py` | Z-score signals, momentum block, plateau stop. |

## `src/regime/`

| File | Description |
|---|---|
| `vix.py` | Block new entries above `vix_entry_block`; resume after consecutive days below `vix_resume`. |
| `earnings.py` | Quarter-end blackout and optional pre-earnings exit on losing positions. |

## `src/backtest/`

| File | Description |
|---|---|
| `engine.py` | Daily loop. Warmup load; trades only `[backtest_start_date, backtest_end_date]`. |
| `portfolio.py` | Positions, NAV, drawdown controls. |
| `costs.py` | Commission, slippage, bid-ask, borrow; profit-to-cost skip. |
| `execution.py` | Simulated fills, next-day retry. |

## `src/metrics/`

`performance.compute` and `reporting.generate_report` (script 04). Charts:
NAV/drawdown, monthly heatmap, exit mix, blocked entries, plus `metrics_summary.txt`.

## `scripts/`

| Script | Description |
|---|---|
| `01_fetch_data.py` | CLI around `fetch_all`. Stages: prices → regime → fundamentals → earnings. |
| `02_build_universe.py` | Clean + monthly universe history. |
| `03_run_backtest.py` | `run_backtest(CONFIG)`. Writes trade log, NAV, pair MTM, blocked entries. |
| `04_generate_report.py` | Charts + metrics summary from those CSVs. Does not re-run the engine. |

CLI flags: [`README.md`](../README.md#script-01-cli).

## `tests/`

Synthetic only. No yfinance, no `data/` parquet.

## `outputs/` (gitignored)

| File | Writer |
|---|---|
| `backtest_results/trade_log.csv` | Script 03 |
| `backtest_results/nav_series.csv` | Script 03 |
| `backtest_results/pair_daily_mtm.csv` | Script 03 |
| `backtest_results/blocked_entries.csv` | Script 03 |
| `data_quality_report.txt` | Script 02 / `clean.py` |
| `report/` | Script 04 |

## Not in the live pipeline

| Path | Status |
|---|---|
| `src/tiering/` | Present on disk. Engine scores → trades directly. Do not import in new code. |
| `src/regime/bollinger.py` | Removed. Do not reintroduce. |
| `src/scrap/`, `working_model/`, `scratch/` | Prototypes / one-off analysis. |
