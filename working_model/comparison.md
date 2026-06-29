# Strategy comparison: main pipeline vs `working_model`

Side-by-side view of the **production-oriented stack** in `src/`, `scripts/`, `data/`, and `src/config.py` versus the **`working_model/`** research harness. Canonical strategy narrative lives in `docs/strategy.md`; where the doc mentions empirical spread percentiles, the implemented signal path uses **formation-window z-score thresholds** wired through `CONFIG` (`entry_zscore`, `take_profit_zscore`, `stop_loss_zscore`) and `src/signals/entry_exit.py`.

---

## Executive summary

Both stacks share one thesis: universe → similarity groups → statistically linked pairs → mean-reverting spread trades with risk overlays and portfolio constraints.

They diverge on **discovery geometry** (correlation-distance k-selection vs standardized-return k-means), **pair filtration** (multi-component composite + Johansen vs Engle–Granger + half-life band), **signals** (entry 1.5 / exit 0.5 / stop 3.5 vs 2 / 0 / 3 defaults in `working_model`), **regime overlays** (VIX and earnings present in main; absent in `working_model`), and **backtest fidelity** (`scripts/` engine with tiered costs and leverage rules vs standalone walk-forward NAV with flat bps friction and segment flattening).

---

## Data

### Main pipeline (`src/data/`, `scripts/`, `data/`)

- **Source**: Project fetch/clean/load layer; parquet under `data/raw` and `data/processed` (not read ad hoc outside loaders).
- **Quality**: Cleaning drops names with excessive missing closes in a trailing window (`max_missing_days`, `data_quality_window` in `src/config.py`).
- **Sector labels**: Uses `data/sector_map.py` as part of universe eligibility.

### working_model (`working_model/price_cache.py`, Yahoo Finance)

- **Source**: Adjusted closes and volumes from Yahoo Finance, keyed by ticker list + fetch range.
- **Cache**: Parquet + small metadata JSON under `working_model/cache/raw/` so reruns reuse local panels; optional **`panel_use_start` / `panel_use_end`** slice the loaded panel without refetch.
- **Fetch vs use**: **`fetch_*_date`** defines download/cache identity; **`backtest_*`** defines the evaluation window passed into the walker.
- Does **not** replicate the repo’s full clean/load contract or mandatory sector-map ingestion for labeling.

---

## Universe

### Main pipeline

- **Cadence**: Reconstituted on the order of **monthly** (`universe_refresh_days` ≈ 21 calendar days in `src/config.py`).
- **Hard filters** (point-in-time): minimum price (**$10**), ADV (**≥ 1M shares**), 30-day average **dollar volume ≥ $25M**, **SPY** log-return correlation over **60** days **< 0.90** (insufficient overlap treated conservatively).
- **Structure**: Targets ~**100** names with floor **60**; requires **sector_map** membership; prefers **≥ 8 distinct subsectors** (soft diversity check).

### working_model

- **Base list**: Large fixed multi-sector ticker basket in `WorkingModelConfig.tickers` plus **SPY** for correlation tests.
- **Point-in-time gates** (applied at each walk-forward segment on the **formation** price/volume window, before clustering): **`min_price`**, **`min_adv`**, **`min_dollar_volume`** with **`liquidity_window`**, and **`max_spy_correlation`** over **`spy_correlation_window`** with **`spy_min_observations`** (`working_model/backtester.py`). Logged counts show drops per gate.
- **Gap vs main**: No dynamic universe sizing/floor logic, **no enforced sector_map/subsector diversification**, and **no project `clean.py` quality gate**—only Yahoo panel + the segment filters above.

---

## Clustering

### Main pipeline (`src/clustering/`)

- **Representation**: Pearson **correlation** on **`clustering_window`** trading days of log returns transformed to distance **d = 1 − ρ**.
- **k selection**: Chooses cluster count via **silhouette** across **`k_min` … `k_max`** (**4–6** in config), **`kmeans_restarts`** (**10**), fixed **`random_seed`**.
- **Refresh**: **`clustering_refresh_days`** (**63**, ~quarterly) decouples clustering cadence from the **60-day** signal window.
- **Pairs**: Within-cluster unordered pairs fed to scoring; singleton clusters yield no pairs.

### working_model (`working_model/pairs_strategy.py`)

- **Representation**: **Simple returns** over the entire **formation** slice (**`formation_days`**, default **252** trading days); each ticker’s vector is **column-standardized**, then **`KMeans`** in that Euclidean space.
- **k selection**: With **`use_silhouette_k_selection = True`**, search **`cluster_k_min` … `cluster_k_max`** and pick the **k** with highest mean silhouette; with **`False`** (current default), fixed **`n_clusters`** (**5**). Fallback **`n_clusters`** applies if the silhouette scan fails when enabled.
- **Refresh**: Driven by the walk-forward **segment length** (**`rescore_freq_trading_days`**, default **21**)—each segment rebuilds clusters on the preceding formation window only.
- **Pairs**: Same within-cluster combination rule as main conceptually.

**Takeaway**: Main favors a shorter explicit correlation-distance window + silhouette **k** (in **\[4, 6\]**) and quarterly-style clustering cadence; `working_model` uses a full-formation simple-return panel and either **optional** silhouette **k** in **\[cluster_k_min, cluster_k_max\]** or a **fixed** **`n_clusters`** (**5** by default). Outputs are still not interchangeable without aligning returns, distance, and cadence.

---

## Scoring

### Main pipeline (`src/scoring/`)

- **Cointegration**: **Johansen**-style workflow with Benjamini–Hochberg-style FDR framing at **`johansen_threshold`** (defaults **0.10**).
- **Composite ranking**: Multiple gated components (**correlation stability**, cointegration, **half-life** [min/max **5 / 20** days], volatility ratio, fundamentals/sector compatibility) under configured weights; pairs below **`min_composite_score`** (e.g. **0.55**) drop off.
- **Selection**: **`finalists_per_cluster`** (typically **top 1 pair per cluster** subject to thresholds); additional gates like **`min_recent_correlation`**.

### working_model (`working_model/pairs_strategy.py`)

- **Cointegration**: **Engle–Granger** per within-cluster candidate; **`p < 0.05`**, sorted by **`p_value`** (no multiple-testing correction in code).
- **Half-life gate**: Hedge ratio via **OLS** on overlapping **levels** prices on the formation slice; spread AR(1) half-life must fall in **`half_life_min_days` … `half_life_max_days`** (**5–20**, matching main’s traded half-life band in spirit).

**Takeaway**: Main does broader statistical and economic vetting before a pair trades; `working_model` is a narrower **cointegration + revert-speed** funnel and can admit more false positives statistically.

---

## Signals

### Main pipeline (`src/signals/`)

- **Spread / z**: Formation-locked hedge ratio / mean / scale with **`signal_window`** (**60** days) discipline and **`formation_window`** (**120**) for locked parameters; **`min_formation_spread_std`** floor on denominator.
- **Thresholds** (defaults in `CONFIG`): **`entry_zscore` = 1.5**, **`take_profit_zscore` = 0.5**, **`stop_loss_zscore` = 3.5**.
- **Extras**: Momentum gate (**`momentum_window`**, **`momentum_threshold`**), **`beta_rebalance_threshold`**, **`time_stop_days`**, pair **stop cooldown** (`pair_stop_cooldown_days`).
- Discrete states (**LONG_SPREAD**, **SHORT_SPREAD**, exits, etc.) emerge from **`get_signal`**.

### working_model (`working_model/backtester.py`)

- **Spread / z**: Hedge ratio frozen from **segment formation**; **lagged** rolling z-score on the extended history so decision day **T** does not use bar-**T** spread inputs.
- **Thresholds**: **`entry_z`** **2.0**, **`exit_z`** **0.0**, **`stop_loss_z`** **3.0**, **`max_holding_multiplier` × half-life** for time-stop.
- **Simplicity**: No momentum filter, beta drift rebalance logic, or named discrete signal enum—long/short/flat coded in the walker.

---

## Regime

### Main pipeline (`src/regime/` + `CONFIG`)

- **VIX**: Blocks new entries above **`vix_entry_block`**; resumes below **`vix_resume`** after **`vix_resume_days`** consecutive tame days.

- **Earnings**: Blackout spans **`earnings_blackout_days_before/after`** around quarter ends.

### working_model

- **No**: VIX, earnings blackout, or other macro/calendar overlays in the active loop (`working_model/backtester.py`).

---

## Backtest

### Main pipeline (`src/backtest/`, `scripts/03_run_backtest.py`, `scripts/04_walkforward.py`, etc.)

- Integrated engine with **`backtest_*_date`**, optional **OOS fraction / `run_oos_only`**, **partial fills**, richer execution assumptions.
- **Costs**: **`commission_per_share`**, liquidity-tiered **`slippage_bps_*`**, price-tiered **`bid_ask_bps_*`**, **`min_profit_to_cost_ratio`**.
- Drawdown-sensitive entry sizing and trims (`drawdown_reduce_*`, `drawdown_halt_*`, recovery rules).

### working_model (`working_model/backtester.py`, `pairs_strategy.__main__`)

- **Walk-forward**: Trailing **`formation_days`** discovery, then **`rescore_freq_trading_days`** live days; repeats through **`backtest_start` … `backtest_end`** without look-ahead in z inputs.
- **Segment hygiene**: Pools not carried into the next segment if dropped on reselection—open risk is flattened at segment end (`walk_forward_backtest` loop).
- **Costs**: Flat **`commission_bps`**, **`slippage_bps`**, **`short_borrow_annual`** accrued daily on short notional; no tiered liquidity spread model.
- **Diagnostics**: Printed summary, NAV / active pairs / pair PnL charts and training-vs-OOS timeline via **`summarize_and_plot`** (outputs like `portfolio_performance.png`).

---

## Portfolios

### Main pipeline (`src/config.py`, `src/backtest/`)

- **Capital**: **`initial_capital` = $100,000**.
- **Exposure**: **`max_gross_leverage`** **2×** NAV; **`cash_buffer_pct`**; caps on **`max_subsector_concentration`** and **`max_same_cluster_pairs`** across concurrent trades.
- **Drawdown overlays**: Reduction and halt mechanics tied to NAV path (see **Backtest**).

### working_model (`WorkingModelConfig`, `walk_forward_backtest`)

- **Capital**: **`initial_capital` = $100,000**.
- **Limits**: **`max_active_pairs` = 5**, **`target_gross_per_pair_pct` = **20%** NAV per candidate gross, **`max_gross_exposure_pct` = 100%** of NAV (combined capacity vs main’s leverage/subsector/cluster rules).
- **Accounting**: Explicit **cash + positions** NAV, daily mark-to-market, borrow drag, and friction on turnover (not aggregated simple mean of pair returns).

**Takeaway**: `working_model` is strong for dollar-NAV realism at the prototype tier; main policy encodes fuller **leverage**, **concentration**, and **drawdown** posture.

---

## Reference parameter snapshot (`working_model` defaults)

Representative **`WorkingModelConfig`** values (adjust in **`working_model/configuration.py`**):

| Topic | Typical default |
|---|---|
| Fetch window | **`2018-01-01` … `2025-09-01`** |
| Backtest window | **`2023-01-01` … `2025-05-04`** |
| Formation | **252** trading days |
| Segment length | **21** trading days (`rescore_freq_trading_days`) |
| Cluster k | Optional silhouette in **[cluster_k_min, cluster_k_max]** or fixed **`n_clusters`** (default **5**) |
| Min cointegration history | **200** days |
| Half-life gate | **5–20** days |
| Entry / exit / stop (z) | **2.0 / 0.0 / 3.0** |
| Costs | commission **1 bps**, slippage **2 bps**, borrow **2%**/year |
| Capacity | **5** active pairs, **20%** gross per pair target, **100%** max gross |

Main defaults are summarized in **`src/config.py`** and **`docs/strategy.md`**; they intentionally include controls (subsector/cluster caps, BH-style Johansen cutoff, tiered friction, leverage, regime gates) absent from **`working_model`**.

---

## When to trust which implementation

Use **`working_model`** for rapid walk-forward experimentation, parquet-backed research panels, NAV-level attribution, and pair-count diagnostics. Use the **`src/` + `scripts/`** stack when you need policy-complete behavior aligned with ARQ specs: universe refresh and sector hygiene, composite scoring depth, regime filters, leveraged portfolio rules, and production-style friction.

To move `working_model` toward parity incrementally—without reintroducing the recently rolled-off “everything strict” overlap—prioritize wiring **sector_map-eligible baskets**, aligning **Johansen**, or selectively adding **regime gates**, rather than stacking every guardrail at once.
