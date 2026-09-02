# Pairs Trading

> Mirrored from a team repository. Original commit history is not preserved.

A systematic, market-neutral pairs trading strategy on a multi-sector S&P-style
universe. The pipeline clusters stocks by recent return similarity, scores
within-cluster pairs, and trades mean-reversion of the log-price spread using a
locked formation z-score.

All tunables live in [`src/config.py`](src/config.py).

---

## Overview

Pairs trading exploits the tendency of economically linked stocks to move
together. When two such stocks temporarily diverge, the strategy goes long the
underperformer, short the outperformer, and exits when the spread reverts (or
hits a stop). Positions are dollar-neutral after the hedge ratio.

This repo is a four-person project built with ARQ. The **strategy** is pairs
trading; ARQ is the organization, not the product name.

- **Implemented pipeline:** [`docs/architecture.md`](docs/architecture.md)
- **Original v2.0 spec** (several sections are not live): [`docs/strategy.md`](docs/strategy.md)
- **Data sources and parquet schemas:** [`docs/data.md`](docs/data.md)
- **Measured results and issues:** [`docs/diagnostics.md`](docs/diagnostics.md)
- **Directory tree:** [`docs/file-structure.md`](docs/file-structure.md)

---

## How it works

1. **Universe** — ~95 candidate names, filtered monthly on price, ADV, dollar
   volume, and SPY correlation.
2. **Clustering** — K-means on a correlation distance matrix (`D = 1 − ρ`);
   only within-cluster pairs are scored.
3. **Scoring** — five-component composite (correlation stability, Johansen
   cointegration, spread half-life, volatility, sector). Half-life is a hard
   gate; surviving pairs get locked formation β, μ, and σ.
4. **Signals** — daily z-score of `s = log A − β_F log B`. Enter when the
   spread is stretched, exit on take-profit, stop, plateau, time stop, or
   pre-earnings exit. VIX can block new entries.
5. **Backtest** — custom daily loop with dollar-neutral sizing, modeled costs,
   and drawdown controls. Script 04 charts the resulting CSVs.

Live rules and contracts: [`docs/architecture.md`](docs/architecture.md).
Knobs: [`src/config.py`](src/config.py). Design history:
[`docs/strategy.md`](docs/strategy.md).

---

## Installation and quick start

**Prerequisites:** Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). Internet
access is required for the first data fetch.

```bash
git clone https://github.com/barrettcosgrove/pairs-trading-project.git
cd pairs-trading-project
uv sync
cp .env.example .env
```

Set `ALPHA_VANTAGE_KEY` in `.env` if you want SPY from Alpha Vantage; otherwise
the fetcher falls back to yfinance.

### Pipeline

Run scripts in numbered order. Scripts 02–04 take no flags; change behavior in
`src/config.py` and re-run.

```bash
# Fetch prices, VIX/SPY, fundamentals, and earnings dates
uv run python scripts/01_fetch_data.py --disable-proxy

# Clean prices and build monthly universe history
uv run python scripts/02_build_universe.py

# Full-calendar backtest (CONFIG.backtest_start_date → backtest_end_date)
uv run python scripts/03_run_backtest.py

# Charts and metrics summary from those CSVs
uv run python scripts/04_generate_report.py
```

Review `outputs/data_quality_report.txt` after script 02, before the backtest.

### Script 01 CLI

| Flag | Purpose |
|---|---|
| `--disable-proxy` | Ignore proxy env vars (use if yfinance returns CONNECT 403) |
| `--years` | Years of price history (default 3.5) |
| `--retries` | Retry attempts per ticker on network failure (default 3) |
| `--stage` | `prices`, `regime`, `fundamentals`, `earnings`, or `all` (repeatable) |
| `--skip-prices` / `--skip-regime` / `--skip-fundamentals` | Skip a stage |
| `--resume` / `--no-resume` | Reuse existing parquets; resume fundamentals checkpoint (default: resume) |
| `--force` | Refetch selected stages even if parquet files exist |
| `--fundamentals-delay` | Seconds between fundamental requests (default 5) |
| `--rate-limit-cooldown` | Seconds to wait after a Yahoo rate-limit (default 900) |

Examples:

```bash
uv run python scripts/01_fetch_data.py --years 4.0 --retries 5
uv run python scripts/01_fetch_data.py --stage regime
uv run python scripts/01_fetch_data.py --stage fundamentals --resume
uv run python scripts/01_fetch_data.py --stage earnings
```

### Tests

```bash
uv run pytest
uv run pytest tests/test_scoring.py
uv run pytest --cov=src --cov-report=term-missing
```

Tests use synthetic data only. They never call yfinance or read `data/`.

---

## Architecture

Raw prices flow through a typed pipeline: fetch and clean → monthly universe →
K-means clusters → composite scoring → formation-z signals and regime filters →
daily backtest → report charts.

```
src/
├── config.py      All tunables
├── data/          fetch, clean, load (only parquet reader)
├── universe/      Monthly hard filters
├── clustering/    Distance matrix, K-means
├── scoring/       Five components + composite
├── signals/       Hedge ratio, spread, z-score entry/exit
├── regime/        VIX and earnings
├── backtest/      Engine, portfolio, costs, execution
└── metrics/       Performance stats and script 04 charts
```

Full pipeline diagram and data contracts:
[`docs/architecture.md`](docs/architecture.md).
What each path does: [`docs/file-structure.md`](docs/file-structure.md).
Sources and parquet schemas: [`docs/data.md`](docs/data.md).

---

## Tech stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pandas | 2.2+ | DataFrames, time series |
| numpy | 1.26+ | Numerical computation |
| scikit-learn | 1.4+ | K-means, silhouette |
| statsmodels | 0.14+ | Johansen, OLS |
| yfinance | 0.2.40+ | Prices and fundamentals |
| pyarrow | 15+ | Parquet I/O |
| matplotlib + seaborn | latest | Report charts (script 04) |
| pytest | 8+ | Unit tests |
| ruff | 0.4+ | Lint |
| uv | latest | Environment |

The backtest is a custom daily loop (not vectorbt / backtrader) so formation
locks, VIX hysteresis, and dollar-neutral sizing stay explicit.

---

## Reporting

[`scripts/04_generate_report.py`](scripts/04_generate_report.py) reads the CSVs
from script 03 and writes `outputs/report/`. It does not re-run the engine.

| File | What it shows |
|---|---|
| `nav_and_drawdown.png` | Portfolio NAV and drawdown from peak |
| `monthly_returns_heatmap.png` | Year-by-month NAV returns |
| `exit_type_mix.png` | Exit counts (take-profit, stop, time stop, …) with net P&L |
| `blocked_entries.png` | Why candidate entries were suppressed |
| `metrics_summary.txt` | Sharpe, drawdown, win rate, and related stats |

Interpretation of runs: [`docs/diagnostics.md`](docs/diagnostics.md).

---

## Contributions

Four people built this. I outlined the project architecture, designed the data
pipeline (`src/data/`, `src/universe/`, `src/config.py`, scripts 01–02), and
collaborated on the K-means clustering with another teammate.

---

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies.*
- Johansen, S. (1988). Statistical Analysis of Cointegration Vectors. *Journal of Economic Dynamics and Control.*
- Engle, R. F., & Granger, C. W. J. (1987). Co-Integration and Error Correction. *Econometrica.*
