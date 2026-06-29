# File Structure

```
arq-pairs-trading/
├── .env.example
├── .gitignore
├── CLAUDE.md
├── README.md
├── file-structure.md
├── project-structure.md
├── pyproject.toml
├── data/
│   ├── sector_map.py
│   ├── processed/              (gitignored)
│   │   └── correlation_matrices/
│   └── raw/                    (gitignored)
├── docs/
│   ├── architecture.md
│   ├── data.md
│   ├── decisions.md
│   ├── file-structure.md
│   ├── git-conventions.md
│   ├── implementation-plan.md
│   ├── open-questions.md
│   ├── project-structure.md
│   └── strategy.md
├── outputs/                    (gitignored)
│   ├── backtest_results/
│   └── report/
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_build_universe.py
│   ├── 03_run_backtest.py
│   ├── 04_walkforward.py
│   └── 05_generate_report.py
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
│   │   ├── bollinger.py
│   │   ├── earnings.py
│   │   └── vix.py
│   ├── scoring/
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
│   ├── tiering/
│   │   ├── assign.py
│   │   └── kpss.py
│   └── universe/
│       └── filter.py
└── tests/
    ├── fixtures/
    ├── test_backtest.py
    ├── test_clustering.py
    ├── test_scoring.py
    └── test_signals.py
```
