# Working model — strategy breakdown

This document lists **every moving part** of the prototype in `working_model/`: what it does, how parameters control it, and how pieces connect. Values shown as **defaults** come from `working_model/configuration.py` unless noted.

---

## Data

**Purpose:** Feed adjusted closes, volumes, and optionally VIX into the walk-forward engine with reproducible caching.

| Component | Role |
|-----------|------|
| **Price panel** | Wide matrix of **adjusted close** prices per ticker; trading calendar is the row index. |
| **Volume panel** | Same shape/calendar as prices for liquidity-based universe gates. |
| **VIX series** | Daily closes for **`vix_ticker`** (`^VIX`) when the VIX filter is enabled; same fetch/cache pattern as other series. |
| **Returns** | Simple daily percentage returns from prices (`pct_change`); used for clustering only on formation slices. |

**Fetch vs “working” window**

- **`fetch_start_date` / `fetch_end_date`** — Bounds passed to Yahoo Finance and used to name parquet cache files (end date follows yfinance’s **exclusive** end convention).
- **`panel_use_start` / `panel_use_end`** — Optional **in-memory slice** after load: narrows rows **without** changing the cached parquet or refetching.
- These three layers are independent: you can download a long history once, then test only a sub-range from disk.

**Caching**

- Prices → parquet keyed by sorted tickers + request dates (`scrap_prices_*`).
- Volumes → separate parquet (`scrap_volumes_*`).
- VIX → separate parquet (`scrap_vix_*`).
- **`--force-refresh`** on the runner refetches and overwrites cache files.

**Moving parameters:** `fetch_start_date`, `fetch_end_date`, `panel_use_start`, `panel_use_end`, `vix_ticker`.

---

## Universe

**Purpose:** Define **who may enter** the clustering and pair-discovery pool at each walk-forward refresh.

**Base universe**

- A **fixed list** of large-cap U.S. names across sectors (`tickers` in configuration — order is not special).
- **SPY** is loaded for correlation filtering but is **not** clustered or paired.

**Point-in-time hard gates** (evaluated on the **formation** price/volume window ending the day **before** each live segment starts)

| Gate | Default meaning |
|------|-----------------|
| **Price floor** | Latest adjusted close in the liquidity lookback must exceed **`min_price`** (USD). |
| **ADV** | Average daily **share** volume over **`liquidity_window`** days must exceed **`min_adv`**. |
| **Dollar volume** | Average **`price × volume`** over **`liquidity_window`** must exceed **`min_dollar_volume`**. |
| **SPY correlation** | Pearson correlation of simple returns vs **`spy_ticker`** over **`spy_correlation_window`** days must be **below** **`max_spy_correlation`**; requires at least **`spy_min_observations`** overlapping returns. |

Tickers failing any gate are dropped for that segment’s discovery only (lists can change segment to segment).

**Moving parameters:** `tickers`, `min_price`, `min_adv`, `min_dollar_volume`, `liquidity_window`, `max_spy_correlation`, `spy_correlation_window`, `spy_min_observations`, `spy_ticker`.

---

## Clustering

**Purpose:** Partition the **eligible** universe each refresh so only **within-cluster** pairs are tested.

**Representation**

- **Simple returns** over the **formation** trading-day window (**`formation_days`** rows).
- Each ticker’s return series is **standardized** (zero mean, unit variance across that window); rows of this matrix are clustered.

**Algorithm**

- **K-means** with **`kmeans_n_init`** restarts and fixed **`kmeans_random_seed`** for repeatability.

**Choosing K**

- **`use_silhouette_k_selection`** toggles this behavior (defaults **`False`** in configuration — fixed **k** without silhouette unless you opt in).
- If **true**: try every integer **k** from **`cluster_k_min`** through **`cluster_k_max`** (capped by number of stocks − 1); pick the **k** with highest **mean silhouette** on the scaled return rows (Euclidean distance consistent with K-means).
- If **false**: always run K-means with **`n_clusters`** clusters (**5** by default). Silhouette scan is skipped.
- If silhouette is on but no valid **k** wins: fall back to **`n_clusters`** (clipped so **k** ≤ number of stocks − 1).

**CLI:** `--use-silhouette-k-selection` / `--no-use-silhouette-k-selection` overrides configuration for that run (omit both to follow **`configuration.py`**).

**Moving parameters:** `formation_days`, `use_silhouette_k_selection`, `cluster_k_min`, `cluster_k_max`, `n_clusters`, `kmeans_n_init`, `kmeans_random_seed`.

---

## Scoring (pair discovery — statistical filters)

**Purpose:** Turn clusters into a **tradable pair catalog** for the upcoming live segment. There is **no** multi-factor composite rank like the main repo; filters are **gates** plus ordering by Engle–Granger **p-value** when listing candidates.

**Within-cluster candidates**

- All unordered pairs inside each cluster.

**Cointegration**

- **Engle–Granger** two-step test on **adjusted close levels** over the formation slice.
- Minimum overlapping days per test: **`min_coint_history`**, capped by available formation length.
- Standard significance rule: **p &lt; 0.05** (no Benjamini–Hochberg in this prototype).
- Passing pairs are sorted by **p-value** (strongest statistical evidence first in the table).

**Hedge ratio and half-life**

- **OLS** of stock A on stock B (levels) on formation data → hedge ratio and spread.
- Half-life from **AR(1)** on spread changes vs lagged spread (mean-reversion speed in trading days).
- Pair is dropped if half-life is outside **[`half_life_min_days`, `half_life_max_days`]**.

**Moving parameters:** `min_coint_history`, `half_life_min_days`, `half_life_max_days`.

---

## Signals

Mean-reversion on the **spread** between **A** and **B**: each spread is daily **flat** / **long spread** / **short spread**. **`beta`** is fixed from formation only; **`z`** drives entries and mean-reversion exits; **time stop** and other closes live under **Risk management** (time stop does **not** use **`z`**).

**Spread** (adj. closes, formation hedge):

```text
S[t] = PA[t] - beta * PB[t]
```

**Rolling `z`** (one EOD value per day per pair; full series rebuilt each walk-forward segment): window **`w`** from formation half-life **`HL`** (days), capped and floored — bad/missing **`HL`** → **`w = 5`**:

```text
w = max(5, min(int(HL), 252))
```

**Lag (no look-ahead):** numerator is **`S[t-1]`**; trailing **`w`**-day mean and sample std of **`S`** are each shifted one more day so **`S[t]`** never enters the same-day signal. Denominator std is floored at **`1e-9`**.

```text
z[t] = ( S[t-1] - mu_lag[t] ) / max( sigma_lag[t], 1e-9 )
```

**Rules** (defaults in parentheses; **`z`** means **`z[t]`** on the decision date):

| Action | Condition |
|--------|-----------|
| Enter **long** spread | `z ≤ -entry_z` (**`entry_z = 2`** → **`z ≤ -2`**) |
| Enter **short** spread | `z ≥ +entry_z` (**`z ≥ 2`**) |
| Exit (was long) | `z ≥ exit_z` (**`exit_z = 0`**) |
| Exit (was short) | `z ≤ exit_z` (**`z ≤ 0`**) |
| **`z` stop-loss** | `|z| ≥ stop_loss_z` (**`stop_loss_z = 3`**) |

New openings can be limited by discovery order, **`max_active_pairs`**, and **`max_gross_exposure_pct`** (**Portfolios** / **Risk**).

**Moving parameters:** `entry_z`, `exit_z`, `stop_loss_z`, `max_holding_multiplier`; **`w`** via **`HL`** and **`half_life_min_days` / `half_life_max_days`**.

---

## Regime

**Purpose:** Optionally suppress **new risk** when implied volatility suggests a stressed regime; **never** blocks risk-reducing exits.

**VIX filter (optional)**

- Controlled by **`use_vix_filter`** (defaults **on** in current `configuration.py`; set **`False`** for always-on entries).
- **Entry rule uses lagged VIX only:** VIX is aligned to the equity calendar, forward-filled across gaps, then **shifted by one trading day**. On date **d**, the gate sees **previous** session’s VIX close — **no same-day VIX peek** alongside the spread logic.
- **Block new entries** when lagged VIX ≥ **`vix_entry_block`** (default **28**).
- **Resume** new entries only after lagged VIX ≤ **`vix_resume`** (default **25**) for **`vix_resume_days`** consecutive trading days (default **5**, hysteresis).
- **Stops, take-profit exits, time-stops, segment drop-outs** still execute when the filter is on.

**Moving parameters:** `use_vix_filter`, `vix_ticker`, `vix_entry_block`, `vix_resume`, `vix_resume_days`.

---

## Risk management

**Purpose:** Summarize how the prototype limits **directional market risk**, **position-level loss**, **concentration**, and **implementation drag** — and what is **not** hedged away.

### Market-neutral design (spread / dollar intuition)

- Each live position is a **two-leg spread**: one leg is **long** and one **short**, with **dollar weights** chosen from the **formation hedge ratio** and a **target gross notional** for the pair. The intent is **near–dollar-neutral** exposure to the two names so **PnL comes mainly from mean reversion of the spread**, not from a single-stock beta bet.
- **Not the same as “beta zero to the S&P”:** the book is **not** explicitly hedged to broad market factors. Residual **equity beta**, **sector**, and **style** exposure can remain, especially if the hedge is estimated with error or the relationship breaks.
- **Universe screening as soft risk control:** **SPY correlation caps**, **liquidity floors**, and **price / dollar-volume** gates reduce names that behave like passive index proxies or are expensive to trade (see **Universe**).

### Stop-loss (z-score risk limit)

- Measured on the **same lagged rolling z-score** used for entries (see **Signals**) — consistent timing, **no separate “faster” path** using future bars.
- If an open spread sees **|**z**|** ≥ **`stop_loss_z`** (default **3.0**), the position is **closed** at that day’s model execution. Exit is counted as a **loss** in the printed win/loss style stats — treat that label together with dollar PnL.
- Interpretation: the spread moved **against** the position **beyond** the band you allow before admitting the thesis failed for that episode.

### Time stop (“time loss” / horizon cap)

- If **calendar days held** exceeds **`max_holding_multiplier`** × **formation-estimated half-life** (default **3.0** × HL; HL at trade time comes from formation and was already required in [**`half_life_min_days`, `half_life_max_days`**] at discovery), the spread is **closed** even if neither mean-reversion exit nor z-stop triggered.
- Rationale: if mean reversion has not kicked in within a **few half-lives**, the model treats the residual as **stale** relative to its own speed benchmark (structural drift, regime change, or bad pair).
- Likewise counted as a **loss** in the coarse exit bookkeeping.

### Mean-reversion exit (profit-taking, not “loss” controls)

- When **|**z**|** reverts toward **`exit_z`** (default crossing **0** from the direction of entry), the spread is closed and counted as a **win** in the exit-classification tally. This is **not** defensive risk in the stop sense, but it **defines when edge is crystallized.**

### Position sizing and exposure caps

- **Per-pair gross target:** each new activation targets **`target_gross_per_pair_pct`** of **current NAV** (default **20%**) as approximate **combined long + short gross** dollar scale for that spread.
- **Book gross cap:** total portfolio **gross** exposure cannot exceed **`max_gross_exposure_pct`** × NAV (default **100%**).
- **Diversification cap:** at most **`max_active_pairs`** (default **5**) spreads **open at once** — limits single-date concentration in a small subset of pairs.
- **Sizing is NAV-linked:** as NAV rises or falls, **dollar notionals** shrink or grow mechanically with the configured percentages (**path risk**: drawdown reduces capacity).

### Costs and financing (risk drag)

- **Commission** and **slippage** (**`commission_bps`**, **`slippage_bps`**) hit **each trade’s** notional; frequent entries/exits compound.
- **Short borrow** (**`short_borrow_annual`**, accrued daily on **short leg dollar exposure** at **`trading_days_per_year`**) is a **persistent drag** while shorts are open — material when gross per pair is high or holds are long.

### Regime filter (entry risk only)

- With **`use_vix_filter`** on, **new** risk is throttled in high-VIX regimes (see **Regime**). **Exits and risk exits are not blocked** — you can still cut losses or take profit.

### Walk-forward / segment discipline (model risk)

- **End-of-segment flattening** forces all positions **closed** before the next discovery segment. That is a **hard reset** of hedge definitions and pair lists — it caps **model staleness** but can realize **turnover and costs**.

### What the summary metrics reflect

- Reported **return, Sharpe, max drawdown** come from the **NAV path** (cash + MTM), not from abstract per-pair averages.
- **Win rate / trade counts** from exit labels should be read **next to** dollar PnL and drawdown — a high “win” label count can still coincide with small edge after costs.

**Primary configuration knobs (risk):** `entry_z`, `exit_z`, `stop_loss_z`, `max_holding_multiplier`, `half_life_min_days`, `half_life_max_days`, `max_active_pairs`, `target_gross_per_pair_pct`, `max_gross_exposure_pct`, `commission_bps`, `slippage_bps`, `short_borrow_annual`, `use_vix_filter`, `vix_entry_block`, `vix_resume`, `vix_resume_days`, `rescore_freq_trading_days` (turnover cadence vs `formation_days`).

---

## Backtest

**Purpose:** Simulate **walk-forward** discovery and trading, produce **out-of-sample** performance metrics and charts in **dollar NAV** space — **without** using future prices or future labels when forming decisions.

### What we are testing

- **Economic question:** Does this pipeline’s combination of universe gates, clustering, cointegration + half-life filters, z-signals, costs, and optional VIX overlay produce acceptable **realized NAV growth** and risk statistics over a chosen **evaluation window**?
- **Statistical question:** Results are **path-dependent** and sensitive to cadence; treat outputs as **research evidence**, not live guarantees.

### Out-of-sample (OOS) definition

- **`backtest_start` / `backtest_end`** — First and last **dates (inclusive)** for which portfolio returns and NAV are reported and plotted as the main experiment window.
- **Training / formation data** still exists **before** `backtest_start`: it is used only for **initial** segments whose formation windows lie partly or wholly pre-OOS; **no future segment’s data** is used to build signals for an earlier date.

### Walk-forward structure (no peeking)

1. **Formation window:** **`formation_days`** trading rows ending the day **before** the segment’s first live day — used **only** for universe gates, clustering, cointegration, hedge ratio, and half-life for that segment’s catalog.
2. **Live segment:** **`rescore_freq_trading_days`** consecutive trading days — signals and portfolio updates run day by day.
3. Repeat: slide forward by one segment length; recompute everything from a **new** trailing formation window.

### Segment discipline

- At segment end, **open positions are flattened** before the next segment so hedge ratios and cluster memberships do not carry conflicting definitions across refreshes.

### Look-ahead bias controls (explicit)

| Area | Protection |
|------|------------|
| Discovery | Formation slices use rows **strictly before** the segment’s first live day. |
| Z-score | Lagged rolling mean/std and lagged spread input so date **t** does not embed same-bar knowledge in the usual sense of the rolling statistic. |
| VIX gate | **Shift(1)** on aligned VIX so the decision at **d** does not use **d**’s VIX close as of the model timing assumed here. |
| Universe gates | Computed from formation-window prices/volumes available before live trading for that segment. |

**Moving parameters:** `backtest_start`, `backtest_end`, `formation_days`, `rescore_freq_trading_days`.

---

## Portfolios

**Purpose:** Translate signals into **cash**, **positions**, and **NAV** with capacity and friction.

**Capital and exposure**

- Starting cash/NAV: **`initial_capital`** (default **$100,000**).
- At most **`max_active_pairs`** simultaneous spread positions.
- Each new position targets **`target_gross_per_pair_pct`** of **current NAV** as gross spread dollars; total gross exposure cannot exceed **`max_gross_exposure_pct`** of NAV.

**Execution model**

- Dollar-neutral intuition: gross dollars split across legs using hedge ratio and **close prices** on the trade day.
- **Commission** and **slippage** apply to traded notional (**`commission_bps`**, **`slippage_bps`**).
- **Short borrow** accrues daily on short leg dollar exposure at **`short_borrow_annual`** over **`trading_days_per_year`**.

**NAV**

- **NAV = cash + mark-to-market value of all open legs** each day.
- Daily portfolio returns from NAV changes feed summary statistics (return, Sharpe, drawdown, etc.).

**Moving parameters:** `initial_capital`, `max_active_pairs`, `target_gross_per_pair_pct`, `max_gross_exposure_pct`, `commission_bps`, `slippage_bps`, `short_borrow_annual`, `trading_days_per_year`.

---

## How to run and how to change parameters

### Run

From the repository root (with `uv` and dependencies installed):

```bash
uv run python working_model/pairs_strategy.py
```

Common flags:

| Flag | Effect |
|------|--------|
| **`--force-refresh`** | Refetch Yahoo data and overwrite parquet in `working_model/cache/raw/`. |
| **`--cache-dir PATH`** | Use a different raw cache directory. |
| **`--use-start` / `--use-end`** | Inclusive date bounds for the **working panel** after load (subset rows; does not change fetch unless you also change fetch dates in configuration). |
| **`--use-vix-filter` / `--no-use-vix-filter`** | Override **`use_vix_filter`** for this run without editing the file. |
| **`--use-silhouette-k-selection` / `--no-use-silhouette-k-selection`** | Override **`use_silhouette_k_selection`** (fixed **`n_clusters`** vs silhouette **k**). |

### Change parameters

- **Single source of defaults:** edit **`working_model/configuration.py`** (`WorkingModelConfig`).
- Frozen dataclass: change values there; no scattered magic numbers in the runner.
- After changing fetch-related bounds, you may need **`--force-refresh`** once if you want new Yahoo ranges written to cache.

### Outputs

- Console summary (OOS window, NAV, return, Sharpe, drawdown, trade counts, pair contributors).
- **`portfolio_performance.png`**, **`portfolio_performance_pair_pnl.png`**, **`portfolio_timeline.png`** (paths configurable in the plotting helper).

---

## Quick reference — all configuration knobs

Grouped by section above:

- **Data:** `fetch_start_date`, `fetch_end_date`, `panel_use_start`, `panel_use_end`, `vix_ticker`
- **Universe:** `tickers`, `min_price`, `min_adv`, `min_dollar_volume`, `liquidity_window`, `max_spy_correlation`, `spy_correlation_window`, `spy_min_observations`, `spy_ticker`
- **Clustering:** `formation_days`, `use_silhouette_k_selection`, `cluster_k_min`, `cluster_k_max`, `n_clusters`, `kmeans_n_init`, `kmeans_random_seed`
- **Scoring:** `min_coint_history`, `half_life_min_days`, `half_life_max_days`
- **Signals:** `entry_z`, `exit_z`, `stop_loss_z`, `max_holding_multiplier`
- **Regime:** `use_vix_filter`, `vix_entry_block`, `vix_resume`, `vix_resume_days`
- **Risk management:** `entry_z`, `exit_z`, `stop_loss_z`, `max_holding_multiplier`, `half_life_min_days`, `half_life_max_days`, plus portfolio caps (`max_active_pairs`, `target_gross_per_pair_pct`, `max_gross_exposure_pct`), friction (`commission_bps`, `slippage_bps`, `short_borrow_annual`, `trading_days_per_year`), and cadence overlap with `formation_days` / `rescore_freq_trading_days`
- **Backtest window:** `backtest_start`, `backtest_end`, `rescore_freq_trading_days`
- **Portfolio:** `initial_capital`, `max_active_pairs`, `target_gross_per_pair_pct`, `max_gross_exposure_pct`, `commission_bps`, `slippage_bps`, `short_borrow_annual`, `trading_days_per_year`

---

## Interpretation notes

- **Win-rate style metrics** reflect exit **labels** (take-profit vs stop vs time-stop); align them with **dollar PnL** and NAV paths.
- **Sensitivity:** Performance moves sharply with formation length, segment length, universe composition, z thresholds, half-life band, silhouette range, and VIX thresholds.
- For fair A/B tests, change **one** knob at a time and keep data/cache alignment identical.
