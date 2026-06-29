# ARQ Pairs Trading Strategy

A systematic, market-neutral pairs trading strategy built on unsupervised machine learning. The strategy identifies historically cointegrated stock pairs within the US sectors, then generates mean-reversion trading signals when the spread between a pair deviates beyond empirical thresholds from its historical distribution.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Testing](#testing)
- [Team and Module Ownership](#team-and-module-ownership)
- [Known Limitations](#known-limitations)
- [Performance Targets](#performance-targets)

---

## Overview

Pairs trading exploits the tendency of economically linked stocks to move together over time. When two such stocks temporarily diverge, a statistical arbitrage opportunity emerges: go long the underperformer, short the outperformer, and profit when they converge back to their historical relationship.

This strategy operationalizes that idea at scale across 100 liquid US technology stocks using a three-stage pipeline:

1. **Clustering** — K-means groups stocks by correlation structure, reducing ~4,950 possible pairs to ~400–600 high-quality candidates
2. **Composite Scoring** — Five quantitative components rank candidates on cointegration strength, reversion speed, volatility compatibility, and fundamental similarity

Trades are entered and exited using empirical percentile thresholds (not normal distribution assumptions) on the spread, filtered by a VIX gate at the portfolio level.

The strategy targets market neutrality (dollar-neutral long/short positions) and mean-reversion horizons of 5–20 days.

---

## How It Works

### Universe Selection
100 liquid technology stocks (GICS Sector 45) that pass hard pre-filters on price (>$10), average daily volume (>1M shares), market cap (>$2B), and SPY correlation (<0.90). Universe is reconstituted monthly.

### Pair Discovery via Clustering
K-means clustering runs on the pairwise correlation distance matrix over a 120-day rolling window. The number of clusters k is selected automatically via silhouette scoring (k ∈ [4, 20]). Only within-cluster pairs proceed to scoring — cross-cluster pairs are never evaluated.

### Composite Score (Five Components)
Each candidate pair is scored on five dimensions and ranked within its cluster:

| Component | Weight | What It Measures |
|---|---|---|
| Correlation Stability | 20% | Ratio of recent 60-day to 2-year historical correlation |
| Johansen Cointegration | 30% | Statistical evidence that the spread is stationary |
| Spread Half-Life | 25% | Days to revert halfway to the mean (target: 5–20 days) |
| Volatility Compatibility | 15% | How well-matched the two stocks' volatilities are |
| Fundamental Compatibility | 10% | P/S ratio and revenue growth similarity |

The top 1 pair per cluster proceeds to trading if it passes the absolute minimum composite score threshold.

### Signal Generation
For each confirmed pair, the hedge ratio β is estimated daily via rolling 60-day OLS regression on log prices. Entry and exit thresholds are based on the empirical percentile distribution of the spread — not a normal distribution assumption — to account for fat tails in tech stock spreads.

| Signal | Condition | Action |
|---|---|---|
| Long spread | Spread < 2nd percentile | Long Stock A, Short Stock B |
| Short spread | Spread > 98th percentile | Short Stock A, Long Stock B |
| Take profit | Spread re-enters 40th–60th percentile | Close position |
| Stop loss | Spread > 99.5th or < 0.5th percentile | Close position |
| Time stop | Position open > 20 trading days | Close position |

### Risk Management
- Dollar-neutral sizing: long and short legs equal in dollar value after β adjustment
- No stock may appear in more than one active pair simultaneously
- Maximum gross leverage: 2.0×
- Drawdown controls: position size reduction at 5% drawdown, full halt at 10%
- VIX filter: no new entries when VIX > 28

---

## Project Structure

```
arq-pairs-trading/
├── README.md                    ← You are here
├── CLAUDE.md                    ← AI assistant context (read before coding)
├── pyproject.toml               ← Python dependencies
├── .env.example                 ← Environment variable template
│
├── docs/                        ← Strategy and architecture documentation
│   ├── strategy.md              ← Full strategy specification (v2.0)
│   ├── architecture.md          ← System design and data flow
│   ├── data.md                  ← Data sources, schemas, known issues
│   ├── implementation-plan.md   ← Two-week build roadmap with owners
│   ├── decisions.md             ← Architecture decision log
│   └── open-questions.md        ← Unresolved items tracker
│
├── data/
│   ├── sector_map.py            ← Ticker → subsector mapping (committed)
│   ├── raw/                     ← Fetched data (gitignored)
│   └── processed/               ← Derived data cache (gitignored)
│
├── src/
│   ├── config.py                ← All tunable parameters (single source of truth)
│   ├── data/                    ← Fetch, clean, load
│   ├── universe/                ← Hard pre-filters
│   ├── clustering/              ← Correlation matrix, K-means
│   ├── scoring/                 ← Five composite score components
│   ├── signals/                 ← Hedge ratio, spread, entry/exit signals
│   ├── regime/                  ← VIX, earnings blackout filters
│   ├── backtest/                ← Simulation engine, portfolio, costs, execution
│   └── metrics/                 ← Performance stats, report generation
│
├── scripts/
│   ├── 01_fetch_data.py         ← Run once to download all data
│   ├── 02_build_universe.py     ← Clean data and apply universe filters
│   ├── 03_run_backtest.py       ← Main backtest entry point
│   ├── 04_walkforward.py        ← Walk-forward validation
│   └── 05_generate_report.py   ← Charts and tables for final report
│
├── tests/                       ← pytest unit tests (run before every PR)
│   ├── fixtures/                ← Synthetic test datasets
│   ├── test_clustering.py
│   ├── test_scoring.py
│   ├── test_signals.py
│   └── test_backtest.py
│
└── outputs/                     ← All generated results (gitignored)
    ├── backtest_results/
    ├── report/
    └── data_quality_report.txt
```

For full descriptions of every file, see [`docs/architecture.md`](docs/architecture.md).

---

## Quick Start

### Prerequisites
- Python 3.11+
- `uv` (recommended) or `poetry` for dependency management
- Internet access for the initial data fetch

### Installation

```bash
# Clone the repo
git clone https://github.com/your-org/arq-pairs-trading.git
cd arq-pairs-trading

# Install dependencies with uv
uv sync

# Or with poetry
poetry install

# Copy environment template (add any API keys if needed)
cp .env.example .env
```

### First-Time Setup

```bash
# Step 1: Fetch all historical data (run once, takes 2–5 minutes)
python scripts/01_fetch_data.py

# Step 2: Clean data and build the universe history
python scripts/02_build_universe.py

# Check data/data_quality_report.txt for any issues before proceeding
```

---

## Running the Pipeline

Run scripts in numbered order. Each script depends on the outputs of the previous one.

```bash
# Full backtest on training period
python scripts/03_run_backtest.py

# Walk-forward validation across quarterly windows
python scripts/04_walkforward.py

# Generate all charts and tables for the report
python scripts/05_generate_report.py
```

Results are written to `outputs/`. The NAV curve, trade log, and per-pair P&L summary are in `outputs/backtest_results/`. Report charts are in `outputs/report/`.

### Sensitivity Analysis

To test alternative parameter sets, pass a config override to the backtest script:

```bash
python scripts/03_run_backtest.py --entry-percentile 5 --exit-percentile 95
```

All available parameters are documented in `src/config.py`.

---

## Configuration

All tunable parameters live in `src/config.py` as a frozen dataclass. **Never hardcode parameter values in module files.** Always import `CONFIG` from `src/config.py`.

Key parameters and their defaults:

| Parameter | Default | Description |
|---|---|---|
| `clustering_window` | 120 days | Rolling window for correlation matrix used in clustering |
| `signal_window` | 60 days | Rolling window for β, μ, σ, and empirical percentiles |
| `k_min / k_max` | 4 / 20 | Range of cluster counts evaluated by silhouette scoring |
| `entry_percentile_low` | 2.0 | Spread percentile threshold for long entry |
| `entry_percentile_high` | 98.0 | Spread percentile threshold for short entry |
| `exit_percentile_low` | 40.0 | Lower bound of take-profit zone |
| `exit_percentile_high` | 60.0 | Upper bound of take-profit zone |
| `min_composite_score` | 0.70 | Minimum threshold for a cluster finalist to be traded |
| `short_borrow_annual` | 0.02 | Flat annual short borrow cost assumption |
| `random_seed` | 42 | Fixed seed for K-means reproducibility |

---

## Testing

Tests use synthetic datasets in `tests/fixtures/` and do not require internet access. Run the full suite before opening any PR.

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/test_scoring.py

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

All tests should pass in under 30 seconds. If a test requires real market data or network access, it belongs in a notebook, not the test suite.

---

## Team and Module Ownership

| Member | Owns | Presentation |
|---|---|---|
| Barrett | `universe.py`, `data.py` | Stock universe, date ranges, liquidity filter |
| Althan | `clustering.py`, assist `pair_selection.py` | Clustering/correlation visuals |
| Anvay | `signals.py`, `backtest.py` | Trading logic and trade simulation results |
| Nanshu | `metrics.py`, `plots.py` | Results charts, help assemble final deck |

Each owner is responsible for the unit tests covering their module.

---

## Known Limitations

The following are intentional simplifications made for the project scope. Each is documented in `docs/data.md` and should be discussed in the final report.

**Survivorship bias** — yfinance only returns currently-listed stocks. Stocks that were delisted or acquired during the backtest period are excluded from the universe. This overstates performance relative to a live strategy that would have held those positions.

**Fundamental data staleness** — P/S ratios and revenue growth figures are used as a current snapshot rather than as true point-in-time historical values. A production implementation would use as-reported data with earnings release date lags.

**Short borrow costs** — A flat 2% annualized assumption is applied to all short positions. Actual borrow rates vary by stock and market conditions and can be significantly higher for heavily-shorted names.

**Earnings blackout** — The blackout window uses end-of-quarter approximation (last 5 trading days of March, June, September, December) rather than per-company earnings dates.

**Multiple testing correction** — Benjamini-Hochberg FDR correction is applied within clusters but not globally across all pairs. A more rigorous implementation would correct globally.

---

## Performance Targets

| Metric | Target |
|---|---|
| Sharpe Ratio (net of costs) | > 1.50 |
| Sortino Ratio | > 2.0 |
| Maximum Drawdown | < 10% |
| Win Rate | 60–70% |
| OOS vs. in-sample deviation | < 30% on all metrics |

Benchmark: Buy-and-hold XLK (Technology Select Sector ETF), equal-weighted and rebalanced monthly.

---

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies.*
- Johansen, S. (1988). Statistical Analysis of Cointegration Vectors. *Journal of Economic Dynamics and Control.*
- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction. *Econometrica.*
