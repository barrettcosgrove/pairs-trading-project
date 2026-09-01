# Strategy Specification

> **Status — original v2.0 design spec, not the live engine.**
>
> Several sections below describe features that were cut or replaced:
> empirical percentiles, pair tiering, KPSS confirmation, Bollinger regime
> gating, P/S fundamentals, tech-only universe, and `k` up to 20.
>
> **What actually runs:** `src/config.py` + the code.
> **Implemented pipeline and contracts:** [`architecture.md`](architecture.md)
> **Measured issues and NAV:** [`diagnostics.md`](diagnostics.md)
>
> Keep this file as design history. Do not “fix the code” to match this
> spec unless a section is explicitly re-adopted.

<!-- Full refined strategy spec (v2.0): universe selection, clustering, composite scoring, tiering, regime filters, signal generation, exit rules, position sizing, and risk management. -->

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Universe Selection](#2-universe-selection)
3. [Clustering](#3-clustering)
4. [Composite Score](#4-composite-score)
5. [Pair Tiering System](#5-pair-tiering-system)
6. [Market Regime Filters](#6-market-regime-filters)
7. [Trading Signals](#7-trading-signals)
8. [Exit Rules](#8-exit-rules)
9. [Position Sizing](#9-position-sizing)
10. [Risk Management](#10-risk-management)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Key Risks](#12-key-risks)

---

## 1. Executive Summary

This spec describes a systematic, market-neutral pairs trading strategy on S&P 500 stocks. The design identifies statistically cointegrated pairs and generates mean-reversion signals using empirical spread percentiles.

**Core thesis.** Highly correlated stocks share structural drivers — revenue mix, customer concentration, macro sensitivity — that prevent their prices from diverging permanently. When a transient dislocation pushes the spread to an extreme percentile, the expected path is reversion to the mean. We capture that reversion by trading dollar-neutral long/short spreads.

**Pipeline in brief:**

```
Raw prices + fundamentals
    → Universe filter (5 hard gates, monthly refresh)
    → K-means clustering (correlation distance matrix, 120-day window)
    → Five-component composite scoring (within each cluster)
    → Tiering by Johansen + KPSS stationarity evidence
    → Regime gating (Bollinger, VIX, earnings blackout)
    → Entry/exit signals (empirical spread percentiles, 60-day window)
    → Dollar-neutral execution with tiered capital pools
    → Performance metrics and walk-forward validation
```

**Capital:** $100,000 starting capital.
**Gross leverage:** capped at 2× NAV.
**Concurrent pairs:** up to *k* pairs (4 to 6).
**Capital allocation:** Equal allocation among active pairs, or dynamic allocation

---

## 2. Universe Selection

Universe construction runs monthly (`universe_refresh_days = 21` calendar days). All five filters are evaluated using only data available on or before the reconstitution date — no look-ahead bias.

**Target size:** 100 tickers (`universe_size`). If filters eliminate too many candidates, the pipeline proceeds with a minimum of 60 (`universe_floor`).

### Filter 1 — Sector

Ticker must appear in `data/sector_map.py`. Only US stocks with a defined subsector are eligible. Any ticker absent from the map is treated as outside the investable universe.

After all filters are applied, the passing tickers must span at least **8 distinct subsectors** (`min_subsectors = 8`). If they do not, a warning is logged and the pipeline proceeds — subsector diversity is a soft check, not a hard gate.

### Filter 2 — Liquidity

30-day average daily volume must exceed **1,000,000 shares** (`min_adv = 1_000_000`). Computed using the 30 most recent trading days of price data as of the reconstitution date.

### Filter 3 — Price

Most recent adjusted close must exceed **$10.00** (`min_price = 10.0`). Eliminates penny stocks where bid-ask spreads are wide and price behavior is erratic.

### Filter 4 — Size (Dollar Volume Proxy)

30-day average daily dollar volume (adj_close × volume) must exceed **$25,000,000** (`min_dollar_volume = 25_000_000`). Used as a size proxy in place of market capitalization — yfinance does not provide shares outstanding in OHLCV data, and a true market cap calculation would require a separate data feed. $25M ADV effectively filters out micro-caps while admitting mid and large caps. Documented as a known approximation.

### Filter 5 — SPY Correlation

60-day Pearson correlation between the ticker's log returns and SPY log returns must be **below 0.90** (`max_spy_correlation = 0.90`). Stocks with correlations above this threshold are effectively index proxies — their spread dynamics are driven by macro, not company-specific factors, making mean-reversion unreliable. Tickers with fewer than 30 aligned observations are conservatively rejected (treated as correlation = 1.0).

### Data Quality Gate

During the cleaning step, any ticker with more than **5 missing adj_close values** (`max_missing_days = 5`) in the trailing **90 calendar days** (`data_quality_window = 90`) is dropped before the universe filter runs.

### Summary

| Filter | Threshold | Parameter |
|---|---|---|
| Sector membership | In `SECTOR_MAP` | — |
| 30-day ADV | > 1,000,000 shares | `min_adv` |
| Price | > $10.00 | `min_price` |
| 30-day avg dollar volume | > $25,000,000 | `min_dollar_volume` |
| 60-day SPY correlation | < 0.90 | `max_spy_correlation` |
| Subsector diversity (soft) | ≥ 8 subsectors | `min_subsectors` |

---

## 3. Clustering

Clustering organizes the investable universe into groups of structurally similar stocks. Only within-cluster pairs are evaluated — cross-cluster pairs are never passed to the scoring module.

### Correlation Distance Matrix

A Pearson correlation matrix is built from the trailing **120 trading days** of log returns (`clustering_window = 120`). This window is intentionally decoupled from `signal_window` (60 days) to avoid in-sample leakage between the clustering step and signal calibration.

The distance matrix is defined as:

```
distance(A, B) = 1 − correlation(A, B)
```

Values range from 0 (perfectly correlated) to 2 (perfectly anti-correlated).

In the current implementation, the trailing window is pivoted into a
date-by-ticker matrix before correlation is computed. Tickers with incomplete
history inside the trailing window are dropped from the clustering snapshot and
logged rather than causing the entire clustering step to fail.

### K-Means Clustering

K-means is run over the distance matrix. The optimal number of clusters `k` is selected by maximizing the silhouette score over the configured range **k ∈ [CONFIG.k_min, CONFIG.k_max]** (currently `4` to `6`). Each candidate `k` is evaluated with **10 random restarts** (`kmeans_restarts = 10`). Random seed is fixed at **42** (`random_seed = 42`) for reproducibility.

Clustering refreshes every **63 calendar days** (`clustering_refresh_days = 63`, approximately quarterly). The implementation currently logs the winning `k` and silhouette score for each run.

### Output

```python
run_clustering(distance_matrix) -> dict[int, list[str]]
# keys: cluster id (0-indexed int)
# values: list of ticker strings in that cluster
```

### Candidate Pair Generation

Cluster assignments are expanded into all unique unordered within-cluster
ticker pairs before scoring. Singleton clusters contribute zero candidate pairs
because they have no partner stock to compare against. Candidate pairs retain
their `cluster_id` so the composite scorer can rank and select finalists
within each cluster.

---

## 4. Composite Score

Each within-cluster pair is evaluated on five components. Components with hard disqualification gates are applied first; pairs that fail any gate are discarded before normalization. Surviving pairs are min-max normalized within the cluster and combined with the weights below. The **top 1 pair per cluster** advances to trading, ONLY if its score > `CONFIG.min_composite_score` (e.g., 0.70).

### Weight Summary

| Component | Weight | Parameter |
|---|---|---|
| Correlation Stability | 20% | `weight_correlation_stability` |
| Cointegration | 30% | `weight_cointegration` |
| Spread Half-Life | 25% | `weight_halflife` |
| Volatility Compatibility | 15% | `weight_volatility` |
| Fundamental Compatibility | 10% | `weight_fundamentals` |

### 4.1 Correlation Stability (20%)

**What it measures.** Whether the pair's recent correlation is consistent with its historical baseline. A pair with a stable, high long-run correlation but a sudden drop in recent correlation is structurally changing — the mean-reversion assumption weakens.

**Computation.** Pearson correlation is computed on aligned pair return series
using two windows:

- Recent window: **60 trading days** (`signal_window = 60`)
- Historical baseline: **252 trading days**
  (`correlation_stability_historical_window = 252`)

The current implementation scores stability as:

```text
score = 1 - abs(recent_correlation - historical_correlation)
```

The result is clipped to `[0, 1]`, so pairs whose recent correlation remains
close to their historical baseline score highest.

**Hard gate.** Pairs where the 60-day Pearson correlation falls below **0.50** (`min_recent_correlation = 0.50`) are discarded before composite scoring regardless of other components.

### 4.2 Cointegration (30%)

**What it measures.** Statistical evidence that the pair's log prices share a long-run equilibrium — the defining requirement for a mean-reverting spread.

**Computation.** Johansen trace test on the pair's log price series. The raw p-value is corrected for multiple testing using Benjamini-Hochberg FDR across all pairs tested within the cluster.

**Hard gate.** Pairs with a BH-corrected p-value ≥ **0.10** (`johansen_threshold = 0.10`) are discarded. For pairs that pass, the score is derived from the corrected p-value: lower p-value → higher score.

### 4.3 Spread Half-Life (25%)

**What it measures.** How quickly the spread mean-reverts. A half-life that is too short produces signals that execute at poor prices (spread flips before fill); too long and capital is tied up too long relative to the time stop.

**Computation.** AR(1) regression of spread change on lagged spread level:

```
Δspread_t = α + β · spread_{t-1} + ε_t
```

Half-life = log(2) / |β|. A negative β (mean-reverting slope) is required; positive slope (diverging) returns `inf`.

**Hard gate.** Pairs with half-life outside **[5, 20] trading days** (`halflife_min = 5`, `halflife_max = 20`) or with a non-negative slope are discarded. The `halflife_max = 20` aligns with `time_stop_days = 20` by design — pairs are selected to be compatible with the time stop duration.

**Scoring.** Within the acceptable range, half-lives closer to the center score higher. The score function favors moderate mean-reversion speed over extreme speeds at either end.

### 4.4 Volatility Compatibility (15%)

**What it measures.** Whether the two stocks have compatible volatility profiles. Large volatility mismatches make dollar-neutral sizing unstable — the hedge ratio must accommodate a large disparity, amplifying leg-specific risk.

**Computation.** Volatility ratio (larger / smaller) computed over two windows:
- Short window: **20 trading days** (`volatility_short_window = 20`)
- Long window: **120 trading days** (`volatility_long_window = 120`)

Combined ratio = 0.60 × short_ratio + 0.40 × long_ratio (`volatility_short_weight = 0.60`, `volatility_long_weight = 0.40`). Lower combined ratio → better score.

**Hard gate.** Pairs where the short-window volatility ratio exceeds **2.50** (`max_volatility_ratio = 2.50`) are discarded before normalization.

### 4.5 Fundamental Compatibility (10%)

**What it measures.** Whether the two stocks share similar business economics. Pairs with fundamentally incompatible valuations or growth profiles are more likely to experience structural divergences that masquerade as mean-reversion opportunities.

**Computation.** Two sub-scores, combined:
1. **P/S ratio similarity** — ratio of the larger P/S to the smaller P/S. Ratios below `max_ps_ratio = 2.50` score well. A mild penalty is applied when both stocks exceed `ps_ratio_high_threshold = 30.0`.
2. **Revenue growth similarity** — absolute difference in TTM revenue growth rate. Differences below `max_revenue_growth_diff = 10.0` percentage points score well.

**Refresh cadence.** Fundamental scores are locked at quarterly boundaries and held constant until the next quarterly refresh. The scoring module does not re-fetch fundamentals within a quarter.

**Data limitation.** P/S ratios and revenue growth are fetched as a current snapshot and held constant over the backtest — not point-in-time. This is a known simplification; a production system would use as-reported data with earnings release date lags.

---

## 5. Market Regime Filters

Three independent filters gate new position entries. All three must permit entry. Existing positions are not force-closed by regime filters — only new entries are blocked.

### 6.1 VIX Filter (Portfolio Level)

**Purpose.** Blocks all new entries when market-wide volatility signals elevated systemic risk, where pairs may experience correlated dislocations that violate the mean-reversion assumption.

**Rules:**
- If VIX ≥ **28.0** (`vix_entry_block`): all new pair entries across the portfolio are blocked immediately.
- Re-entry resumes only after VIX falls below **25.0** (`vix_resume`) and **holds below 25.0 for 5 consecutive trading days** (`vix_resume_days = 5`). This hysteresis prevents whipsawing around the block threshold.

### 6.2 Earnings Blackout (Pair Level)

**Purpose.** Avoids entering positions immediately before or after earnings releases, when idiosyncratic price moves can cause temporary dislocations that are not mean-reverting.

**Rules:**
- No new entries from **5 trading days before** (`earnings_blackout_days_before = 5`) the end of a fiscal quarter through **1 trading day after** (`earnings_blackout_days_after = 1`). The blackout applies to either leg — if either stock in the pair is approaching its earnings window, the pair is blocked.

**Scope note.** Earnings blackout is a scope-cut candidate. See `docs/decisions.md` for the current status.

---

## 7. Trading Signals

### Spread Construction

All signal computation uses **log prices**, not raw prices. The spread is defined as:

```
spread_t = log(price_A_t) − β_t × log(price_B_t)
```

where β_t is the rolling OLS hedge ratio estimated over a **60-day window** (`signal_window = 60`).

**Critical invariant.** β, the spread mean (μ), the spread standard deviation (σ), and the empirical percentile distribution must all use the same `signal_window = 60`. Mixing windows produces miscalibrated thresholds — a silent bug that generates plausible-looking but incorrect signals.

### Hedge Ratio

The hedge ratio β is estimated by OLS regression of log(price_A) on log(price_B) over the trailing 60 trading days. It is recalculated daily with a rolling window.

**Beta rebalance trigger.** If the current β has drifted more than **0.15** (`beta_rebalance_threshold = 0.15`) from the β at which the position was entered, the short leg is resized to restore dollar neutrality.

### Entry Signals

Thresholds are empirical percentiles of the spread distribution computed over the trailing `signal_window = 60` days. This avoids assuming the spread is normally distributed.

| Signal | Condition | Action |
|---|---|---|
| `LONG_SPREAD` | Spread ≤ 2nd percentile (`entry_percentile_low = 2.0`) | Long A, Short B |
| `SHORT_SPREAD` | Spread ≥ 98th percentile (`entry_percentile_high = 98.0`) | Short A, Long B |

Entries are only generated if all regime filters permit (Section 6) and the minimum profit-to-cost ratio is satisfied (Section 9).

---

## 7. Exit Rules

Four independent exit rules are evaluated daily. The first triggered rule closes the position. Rules are checked in priority order: stop-loss → time stop → take profit → hold.

### Take Profit

Close position when the spread re-enters the **[40th, 60th] percentile band** (`exit_percentile_low = 40.0`, `exit_percentile_high = 60.0`). Signal: `TAKE_PROFIT`.

The take-profit band is wide by design. The spread need not return exactly to the mean — a return to the middle quartiles is sufficient to capture the bulk of the reversion while avoiding over-optimization.

### Stop Loss

Close position immediately when the spread reaches an extreme tail:
- **Normal regime:** spread ≤ 0.5th percentile or ≥ 99.5th percentile (`stop_percentile_low = 0.5`, `stop_percentile_high = 99.5`). Signal: `STOP_LOSS`.
- **Bollinger expanded regime:** thresholds tighten to ≤ 1.0th percentile or ≥ 99.0th percentile (`stop_percentile_low_expanded = 1.0`, `stop_percentile_high_expanded = 99.0`). The expanded regime indicates the spread is unusually volatile — a tighter stop limits damage.

### Time Stop

Close position if it has been open for **20 trading days** (`time_stop_days = 20`) without triggering take profit or stop loss. Signal: `TIME_STOP`.

The `time_stop_days = 20` is intentionally matched to `halflife_max = 20`. Pairs are selected because their spread is expected to mean-revert within 20 days. If it has not reverted by then, the pair's statistical profile has likely changed.

### Signal Priority

```
STOP_LOSS > TIME_STOP > TAKE_PROFIT > HOLD
```

If both stop-loss and time-stop conditions hold simultaneously, stop-loss takes precedence (faster exit).

---

## 9. Position Sizing

### Dollar Neutrality

Every pair trade is sized to be dollar-neutral at entry. The long leg and the short leg have equal dollar exposure:

```
size_A = N shares of A  →  dollar_value = N × price_A
size_B = β × N shares of B  →  dollar_value = β × N × price_B ≈ dollar_value_A
```

Dollar neutrality is maintained dynamically via the beta rebalance trigger (Section 7).

### Per-Pair Allocation

Capital is allocated from the appropriate tier pool divided equally among the active pairs in that pool. No pair receives a disproportionate allocation — all slots in a pool are equal-weighted.

- **High-tier pair size** = (60% × NAV) / active_high_pairs, capped at `max_high_pairs = 6` pairs.
- **Low-tier pair size** = (30% × NAV) / active_low_pairs, capped at `max_low_pairs = 8` pairs.

### Concentration Limits

| Constraint | Limit | Parameter |
|---|---|---|
| Max gross leverage | 2.0× NAV | `max_gross_leverage` |
| Max subsector concentration | 35% of active pairs | `max_subsector_concentration` |
| Max pairs from same cluster | 2 | `max_same_cluster_pairs` |

### Minimum Profit-to-Cost Ratio

A trade is only entered if the expected profit (based on spread reversion to the mean) is at least **2.0×** the estimated round-trip transaction cost (`min_profit_to_cost_ratio = 2.0`). This gate prevents entering marginal trades where costs consume most of the edge.

### Transaction Costs

Costs are deducted on every trade. The model includes:

| Cost | Rate | Condition |
|---|---|---|
| Commission | $0.005 per share | Always (`commission_per_share`) |
| Slippage (liquid) | 5 bps | ADV > 5,000,000 shares (`slippage_bps_liquid`) |
| Slippage (medium) | 10 bps | ADV 1,000,000–5,000,000 shares (`slippage_bps_medium`) |
| Bid-ask spread | 2 bps | Price > $30 (`bid_ask_bps_high_price`) |
| Bid-ask spread | 5 bps | Price $10–$30 (`bid_ask_bps_low_price`) |
| Short borrow | 2.0% annually | All short legs (`short_borrow_annual`) |

**Partial fills.** If the short leg fills below **95%** of the intended quantity (`partial_fill_threshold = 0.95`), the long leg order is cancelled. This avoids entering an unhedged long position.

---

## 9. Risk Management

### Drawdown Controls

Two drawdown thresholds trigger position-level responses. Drawdown is measured from the rolling peak NAV.

**Soft threshold — 5% drawdown** (`drawdown_reduce_threshold = 0.05`):
- All new position entries are sized at **50%** of normal (`drawdown_reduce_factor = 0.50`).
- Existing positions are not affected.
- Condition: portfolio is down ≥ 5% from the prior week's close.

**Hard threshold — 10% drawdown** (`drawdown_halt_threshold = 0.10`):
- All new entries are halted.
- Existing positions are trimmed to **25%** of their current size (`drawdown_trim_factor = 0.25`), executed over **5 trading days** (`drawdown_trim_days = 5`) to avoid market impact.
- Recovery: normal operations resume after the portfolio recovers to within 5% of the triggering peak (`drawdown_recovery_threshold = 0.05`) and holds there for **5 consecutive positive trading days** (`drawdown_recovery_days = 5`).

### Gross Leverage Cap

Gross leverage (long exposure + short exposure / NAV) is capped at **2.0×** (`max_gross_leverage = 2.0`). No new entries are permitted if this cap would be breached.

### Subsector Concentration

No more than **35%** of active pairs (`max_subsector_concentration = 0.35`) can share the same technology subsector. This prevents cluster-level correlation from creating a disguised concentrated bet.

### Cluster Concentration

No more than **2 pairs** (`max_same_cluster_pairs = 2`) from the same K-means cluster can be active simultaneously. Pairs from the same cluster share structural drivers — loading more than 2 increases correlated drawdown risk.

---

## 11. Implementation Roadmap

The pipeline is implemented in numbered scripts that must be run in order. Each script depends on the outputs of the previous.

```
scripts/01_fetch_data.py      — Download raw prices, fundamentals, regime data
scripts/02_build_universe.py  — Clean data, apply filters, build universe history
scripts/03_run_backtest.py    — Full simulation on the training period
scripts/04_walkforward.py     — Walk-forward validation
scripts/05_generate_report.py — All performance charts and tables
```

> **Not live.** The implemented pipeline is 01 → 02 → 03 → `04_generate_report.py`.
> Walk-forward was cut; see [`architecture.md`](architecture.md) and [`decisions.md`](decisions.md).
> Install and CLI: [`README.md`](../README.md).

**Out-of-sample split.** The final **20%** of the dataset (`oos_fraction = 0.20`, approximately 6 months of a 2.5-year dataset) is held out and never used for parameter selection or training. Walk-forward validation runs over the training period only; OOS performance is reported separately.

### Scope Cuts (if behind schedule)

Cut in this priority order. Every cut must be logged in `docs/decisions.md`.

| If behind by | Cut | Never cut |
|---|---|---|
| 1 day | Earnings blackout filter | Transaction costs |
| 2 days | Walk-forward validation | OOS held-out period |
| 3 days | Fundamental compatibility component | Johansen + KPSS tiering |
| 4+ days | Pair tiering (use single pool) | Dollar-neutral sizing |

---

## 12. Key Risks

### Survivorship Bias

yfinance only returns currently-listed stocks. Tickers delisted or acquired during the backtest window are absent from the universe, overstating backtest performance. Magnitude is difficult to quantify without a survivorship-free data source. Documented in `docs/data.md`.

### Non-Point-in-Time Fundamentals

P/S ratios and revenue growth are fetched as a current snapshot and applied retroactively across the backtest. A production system would use as-reported data with earnings release date lags to avoid look-ahead bias in the fundamental compatibility component.

### Regime Synthetic Data

During development, `data/raw/regime.parquet` may contain synthetic VIX and SPY data. Synthetic data has exactly as many rows as business days in the date range; real data has slightly fewer due to market holidays. Verify before running a final backtest by checking the row count.

### Window Inconsistency (Silent Bug)

β, μ, σ, and the empirical percentile distribution must all use `signal_window = 60`. If any one of them uses a different window, entry percentiles are miscalibrated against the wrong distribution. This produces plausible-looking but incorrect signals. The code validates this at runtime via `assert` in config validation.

### Cointegration Instability

Johansen tests are sensitive to structural breaks. A pair that was cointegrated over the training period may break down during the OOS period. The monthly retest cadence and the 2-consecutive-failure removal rule are the primary defenses, but they lag the break by up to 2 months.

### Short Borrow Cost Assumption

The flat 2% annual borrow rate (`short_borrow_annual = 0.02`) is a simplification. Heavily shorted tech names can carry borrow rates of 5–30% annually, materially eroding edge. Replace with per-stock broker rates before any live deployment.

### Execution Gap

Signals are generated using market-on-close prices. Fills are simulated at those prices plus the slippage model. In live trading, MOC orders may receive worse fills during volatile periods, and the slippage model may understate true market impact for less-liquid names.

### Gross Leverage Limit

The 2× gross leverage cap (`max_gross_leverage = 2.0`) does not account for intraday margin requirements. During a period where both legs move adversely before close, intraday gross exposure can temporarily exceed 2×. This is a known limitation of a daily-resolution simulation engine.
