# ARQ Pairs Trading Strategy

A systematic, market-neutral pairs trading strategy. The pipeline clusters a
multi-sector S&P-style universe, scores within-cluster pairs, and trades
mean-reversion of the log-price spread using a locked formation z-score.

**Live knobs:** `src/config.py`
**Original v2.0 spec (several sections are not implemented):** `docs/strategy.md`
**Implemented design:** `docs/architecture.md`
**Backtest issues and measured results:** `docs/diagnostics.md`

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

Pairs trading exploits the tendency of economically linked stocks to move
together. When two such stocks temporarily diverge, the strategy goes long
the underperformer, short the outperformer, and exits when the spread
reverts (or hits a stop).

The live pipeline:

1. **Universe** — ~95 multi-sector names from `CANDIDATE_TICKERS`, filtered
   monthly on price, ADV, dollar volume, and SPY correlation
2. **Clustering** — K-means on a 120-day correlation distance matrix
   (`k ∈ [4, 6]`), refreshed about every 63 calendar days
3. **Scoring** — five-component composite; half-life is a hard gate; Johansen
   is a continuous score with a soft floor; exactly one finalist per cluster
   if it clears `min_composite_score`
4. **Signals** — formation z-score of `s = log A − β_F log B` (not empirical
   percentiles). VIX and quarter-end earnings blackout filter new entries

Positions are dollar-neutral after the hedge ratio. Target half-life is
5–20 trading days; the time stop is 50 days.

---

## How It Works

### Universe Selection

Candidate list is hardcoded in `src/data/fetch.py` (~95 S&P-style names
across semiconductors, software, energy, financials, healthcare, staples,
industrials, utilities, materials, consumer discretionary, and
communication services). Monthly hard filters: price > $10, ADV > 1M
shares, 30-day average dollar volume > $25M, 60-day SPY correlation < 0.90.
Must be present in `data/sector_map.py`.

### Pair Discovery via Clustering

K-means on `D = 1 − ρ` over `CONFIG.clustering_window` (120 days). `k` is
chosen by silhouette score in `[k_min, k_max]` = [4, 6]. Only
within-cluster pairs are scored.

### Composite Score (Five Components)

| Component | Weight | What it measures |
|---|---|---|
| Correlation stability | 20% | Recent vs 252-day correlation (0 if recent ρ < 0.50) |
| Johansen cointegration | 25% | `1 −` BH-adjusted p on a 252-day window; soft floor 0.40 |
| Spread half-life | 30% | AR(1) days to revert halfway; **hard gate** [5, 20] |
| Volatility compatibility | 15% | Dual-window vol ratio; discard if short-window ratio > 2.5 |
| Sector compatibility | 10% | Same sector 1.0 / cross-sector 0.4 / unknown 0.5 |

The scorer does **not** use P/S or revenue growth. Those fields are still
fetched into `data/raw/fundamentals.parquet` but are unused.

Pairs with formation β ≤ 0 are dropped. The top 1 pair per cluster proceeds
if `composite_score >= 0.55`.

### Signal Generation

Formation β, μ, and σ are locked at scoring (`formation_window` = 120).
Daily z is `(s_t − μ_F) / σ_F`. A separate 60-day live β (`signal_window`)
only resizes the short leg when it drifts more than 15%.

| Signal | Condition | Action |
|---|---|---|
| Long spread | z < −1.5 | Long A, short B |
| Short spread | z > +1.5 | Short A, long B |
| Take profit | \|z\| < 0.5 | Close |
| Stop loss | \|z\| > 3.5 | Close |
| Time stop | Position open > 50 trading days | Close |
| Momentum block | Either leg moved > 15% in 14 days | No new entry |

### Risk Management

- Dollar-neutral sizing after β
- No ticker in more than one active pair
- Max gross leverage 2.0×; 10% cash buffer
- Drawdown: size-down at 5%, halt at 10%
- VIX: no new entries above 28; resume after 5 days below 25
- 20-trading-day cooldown after a stop on the same pair
- Empty scoring dates keep the last active pair list (do not flatten)

---

## Project Structure

```
arq-pairs-trading/
├── README.md                    ← You are here
├── CLAUDE.md / AGENTS.md        ← AI assistant context
├── pyproject.toml
│
├── docs/
│   ├── architecture.md          ← Implemented pipeline and contracts
│   ├── strategy.md              ← Original v2.0 spec (not all live)
│   ├── diagnostics.md           ← Issues, fixes, backtest results
│   ├── data.md                  ← Sources and parquet schemas
│   ├── file-structure.md        ← Canonical directory tree
│   └── project-structure.md     ← File-by-file inventory
│
├── data/
│   ├── sector_map.py            ← Committed ticker → sector labels
│   ├── raw/                     ← Gitignored fetch output
│   └── processed/               ← Gitignored derived cache
│
├── src/
│   ├── config.py                ← All tunables
│   ├── data/                    ← fetch, clean, load
│   ├── universe/                ← Monthly hard filters
│   ├── clustering/              ← Distance matrix, K-means
│   ├── scoring/                 ← Five components + composite
│   ├── signals/                 ← Hedge ratio, spread, z-score signals
│   ├── regime/                  ← VIX, earnings blackout
│   ├── backtest/                ← Engine, portfolio, costs, execution
│   └── metrics/                 ← Stubs (script 05 not implemented)
│
├── scripts/                     ← 01 fetch → 02 universe → 03 backtest → 04 OOS slice
├── tests/                       ← pytest, synthetic data only
└── outputs/                     ← Gitignored results
```

Full tree: [`docs/file-structure.md`](docs/file-structure.md).
File descriptions: [`docs/project-structure.md`](docs/project-structure.md).

---

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` (recommended)
- Internet access for the initial data fetch

### Installation

```bash
git clone https://github.com/your-org/arq-pairs-trading.git
cd arq-pairs-trading
uv sync
cp .env.example .env
```

Set `ALPHA_VANTAGE_KEY` in `.env` if you want SPY from Alpha Vantage;
otherwise the fetcher falls back to yfinance.

### First-Time Setup

```bash
uv run python scripts/01_fetch_data.py --disable-proxy
uv run python scripts/02_build_universe.py
```

Review `outputs/data_quality_report.txt` before running the backtest.

---

## Running the Pipeline

Run scripts in numbered order.

```bash
# Full-calendar backtest (CONFIG.backtest_start_date → backtest_end_date)
uv run python scripts/03_run_backtest.py

# Slice the last CONFIG.oos_fraction of those CSVs (does not re-fit)
uv run python scripts/04_walkforward.py
```

`scripts/05_generate_report.py` is a stub. There is no generated report yet.

Results:

| File | Writer |
|---|---|
| `outputs/backtest_results/trade_log.csv` | Script 03 |
| `outputs/backtest_results/nav_series.csv` | Script 03 |
| `outputs/backtest_results/pair_daily_mtm.csv` | Script 03 |
| `outputs/backtest_results/oos_*.csv` | Script 04 |

Script 03 has no CLI parameter overrides. Change values in `src/config.py`
(or construct a `StrategyConfig` in code) and re-run.

---

## Configuration

All tunables live in `src/config.py` as a frozen dataclass. Import `CONFIG`.
Never hardcode parameters in module files.

| Parameter | Default | Description |
|---|---|---|
| `clustering_window` | 120 | Correlation matrix for K-means |
| `formation_window` | 120 | Locked β / μ / σ used for z-score |
| `signal_window` | 60 | Live hedge β for share resize only |
| `johansen_window` | 252 | Johansen lookback |
| `k_min` / `k_max` | 4 / 6 | Silhouette-scored cluster count |
| `entry_zscore` | 1.5 | Absolute z to enter |
| `take_profit_zscore` | 0.5 | Absolute z to take profit |
| `stop_loss_zscore` | 3.5 | Absolute z to stop |
| `time_stop_days` | 50 | Force close after this many trading days |
| `min_composite_score` | 0.55 | Floor for a cluster finalist |
| `min_cointegration_score` | 0.40 | Soft floor on `1 − p_adj` |
| `min_formation_beta` | 0.0 | Drop non-positive formation β |
| `backtest_start_date` | 2022-01-01 | First simulated trading day |
| `backtest_end_date` | 2024-12-31 | Last simulated trading day |
| `oos_fraction` | 0.30 | Trailing slice used by script 04 |
| `initial_capital` | 100000 | Starting NAV |
| `random_seed` | 42 | K-means reproducibility |

---

## Testing

```bash
uv run pytest
uv run pytest tests/test_scoring.py
uv run pytest --cov=src --cov-report=term-missing
```

Tests build synthetic frames in-process. They never call yfinance or read
`data/`. The suite should finish in under 30 seconds.

---

## Team and Module Ownership

| Member | Owns |
|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py`, `data/sector_map.py`, `scripts/01`, `scripts/02` |
| Althan | `src/clustering/`, `src/scoring/` |
| Anvay | `src/signals/`, `src/regime/`, `src/backtest/` |
| Nanshu | `src/metrics/`, `scripts/03`, `scripts/04`, `scripts/05` |

Cross-module changes need a PR reviewed by the other owner.

---

## Known Limitations

**Survivorship bias** — yfinance only returns currently listed names.

**Fundamentals snapshot unused** — P/S and TTM growth are fetched once and
are not point-in-time. Live scoring uses sector labels only.

**Short borrow** — Flat 2% annualized on every short leg.

**Earnings blackout** — Last 5 trading days of each calendar quarter, not
per-name earnings dates.

**Script 04 is a slice** — It does not re-cluster or re-score on rolling
windows. OOS is the last 30% of the same continuous simulation.

**Script 05 / metrics** — Not implemented.

Empirical results and open issues: [`docs/diagnostics.md`](docs/diagnostics.md).

---

## Performance Targets

| Metric | Target |
|---|---|
| Sharpe (net of costs) | > 1.50 |
| Sortino | > 2.0 |
| Maximum drawdown | < 10% |
| Win rate | 60–70% |
| OOS vs in-sample deviation | < 30% on all metrics |

These are project goals, not measured results. See `docs/diagnostics.md`
for actual NAV and trade stats.

---

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies.*
- Johansen, S. (1988). Statistical Analysis of Cointegration Vectors. *Journal of Economic Dynamics and Control.*
- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction. *Econometrica.*
