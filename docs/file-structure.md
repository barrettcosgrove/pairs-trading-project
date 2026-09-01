# ARQ Pairs Trading — File Structure

Canonical tree of the repository as it exists now. Generated data, virtual
environments, caches, and report outputs are gitignored and regenerated locally.

**Live behavior** is `src/config.py` plus the code. Original product spec:
`docs/strategy.md`. Empirical issues and backtest rounds: `docs/diagnostics.md`.

```text
arq-pairs-trading/
├── .env.example
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── description.md
├── file-structure.md              # short pointer → this file
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
│   │   ├── fetch_manifest.json
│   │   └── fundamentals_checkpoint.parquet
│   └── processed/                 # gitignored
│       ├── returns.parquet
│       ├── universe_history.parquet
│       └── correlation_matrices/  # optional monthly D = 1−ρ cache
│
├── docs/
│   ├── architecture.md
│   ├── data.md
│   ├── decisions.md
│   ├── diagnostics.md
│   ├── file-structure.md          # this file
│   ├── git-conventions.md
│   ├── implementation-plan.md
│   ├── llm_guide.md
│   ├── module_checlist.md
│   ├── open-questions.md
│   ├── project-structure.md
│   └── strategy.md
│
├── outputs/                       # gitignored
│   ├── backtest_results/
│   │   ├── trade_log.csv
│   │   ├── nav_series.csv
│   │   ├── pair_daily_mtm.csv
│   │   ├── oos_trade_log.csv
│   │   ├── oos_nav_series.csv
│   │   └── oos_pair_daily_mtm.csv
│   ├── report/                    # unused until script 05 is implemented
│   └── data_quality_report.txt
│
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_build_universe.py
│   ├── 03_run_backtest.py
│   ├── 04_walkforward.py
│   └── 05_generate_report.py      # stub
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
│   │   ├── performance.py         # stub
│   │   └── reporting.py           # stub
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
│   │   ├── __init__.py
│   │   └── assign.py
│   ├── scrap/                     # old prototypes; not the live pipeline
│   └── universe/
│       └── filter.py
│
├── tests/
│   ├── fixtures/
│   ├── test_backtest.py
│   ├── test_clustering.py
│   ├── test_cointegration.py
│   ├── test_composite.py
│   ├── test_correlation_stability.py
│   ├── test_halflife.py
│   ├── test_scoring.py
│   ├── test_scoring_integration.py
│   ├── test_signals.py
│   └── test_volatility.py
│
├── scratch/                       # ad-hoc analysis scripts
└── working_model/                 # earlier prototype; not wired to scripts/
```

## Not in the live pipeline

| Path | Status |
|---|---|
| `src/tiering/` | Present on disk. Engine scores → trades directly. Do not import in new code. |
| `src/regime/bollinger.py` | Removed. Do not reintroduce. |
| `src/metrics/performance.py`, `reporting.py` | Header stubs only. |
| `scripts/05_generate_report.py` | Header stub only. |
| `src/scrap/`, `working_model/`, `scratch/` | Prototypes / one-off analysis. |

File-by-file descriptions: [`docs/project-structure.md`](project-structure.md).
