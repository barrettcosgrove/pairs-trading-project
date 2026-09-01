# ARQ Pairs Trading — High-level overview

This note describes what the strategy does, in order, and the **numbers** we use to make decisions. All default values live in a single central configuration in the codebase; if those defaults change, this document should be updated to match.

A few terms used throughout:

- **Log price** — natural logarithm of adjusted closing price. It is standard for spreads and cointegration because percentage moves are closer to additive.
- **Spread** — a single series built from two stocks: how far apart their log prices are after scaling one leg by a **hedge ratio** (often denoted **β**, beta): spread = log(price of A) minus β times log(price of B).
- **Cointegration** — a statistical relationship where two price series drift together over time so their spread can be mean-reverting; we use a formal test (Johansen) with multiple testing correction within each cluster.
- **Z-score** — how many standard deviations today’s spread sits above or below a **formation** mean, using a **formation** standard deviation estimated from history.
- **Dollar-neutral** — long exposure in one stock is offset by short exposure in the other so net dollar exposure is controlled at entry.

---

## What we are trying to do

We trade **pairs** of stocks that look statistically similar and mean-reverting. We only consider pairs **inside the same cluster** (stocks grouped by recent return similarity). For each cluster we pick **one** finalist pair. We trade that pair when its spread is stretched (by z-score), and we exit when it reverts, hits a stop, or hits a time limit. **VIX** and a simple **earnings calendar** can block new risk. **Transaction costs** and a **profit-to-cost** rule can block marginal opens.

---

## Data and simulation window

The backtest pulls **adjusted closes and volume** from `data/raw/prices.parquet` (via the shared data loader), **daily log returns** from `data/processed/returns.parquet`, and **VIX** from `data/raw/regime.parquet` for the volatility regime filter.

The default historical simulation runs from **2019-01-01** through **2024-12-31** (these are configurable). The main simulation runs one continuous timeline. Reporting charts come from `scripts/04_generate_report.py` after the backtest CSVs exist.

---

## Universe — which stocks qualify

Roughly **100** names are targeted, with a floor of **60** if fewer pass filters.

| Criterion | Level |
|-----------|--------|
| Minimum stock price | **$10** |
| Minimum average daily volume | **1,000,000** shares |
| Minimum 30-day average dollar volume | **$25,000,000** |
| Maximum 60-day correlation of log returns to SPY | **0.90** |
| Minimum number of distinct tech subsectors in the set | **8** |
| Data quality: missing prices in a **90**-day lookback | at most **5** days |
| How often the universe is refreshed in the simulation | about every **21** calendar days |

The investable universe is also restricted to tickers listed in **`data/sector_map.py`** (anything not mapped is screened out upstream when `universe_history` is built).

## Clustering — how we group stocks

We measure similarity with **120** trading days of daily **log returns**. Correlation **ρ** between two stocks becomes a distance **1 − ρ** (closer stocks have smaller distance).

When building correlations, the rolling window requests roughly **40%** extra calendar slack so holiday gaps do not starve the trailing **120** trading-day requirement.

For each candidate **k** between **4** and **6**, scikit-learn fits K-means with **`kmeans_restarts` = 10** centroid initializations (fixed seed **42**), then retains whichever **k** delivered the best **silhouette** score. (**`ari_stability_threshold` = 0.50** exists in configuration today but is **not read** by the live clustering path—keep it in mind only if you extend stability monitoring.)

Clusters are rebuilt whenever **63** calendar days elapse since the last successful clustering pass. A thrown exception skips updating that clock, so the engine **retries every subsequent session** until the pipeline completes (only hard failures wedge you off calendar). Successful runs—even when composite scoring yields **zero** finalists—still advance the clustering clock and clear finalist slots until the next quarterly cycle.

---

## Scoring pairs — how we pick one pair per cluster

Every unordered pair inside a cluster is a **candidate**. Each pair gets component scores between **0** and **1**, then a **composite score** as a weighted sum (no rescaling across clusters):

| Component | Weight |
|-----------|--------|
| Correlation stability | **20%** |
| Cointegration (Johansen, adjusted for multiple tests within the cluster) | **30%** |
| Half-life of mean reversion | **25%** |
| Volatility compatibility | **15%** |
| Fundamentals (sector compatibility from a static map) | **10%** |

**Minimum composite score** to stay in contention: **0.55**.

**Hard gates** (applied after component scores exist — these **drop pairs entirely**):
- Spread **half-life** must map to **`halflife_score` ≠ 0** — outside **[5, 20]** trading days (or insufficient data) ⇒ pair removed.
- **Johansen / cointegration** — pairs that fail the Benjamini–Hochberg screening within their cluster (`cointegration_score == 0`) are removed.

**Soft component behavior** (does **not** use the halflife/Johansen hard gate machinery; instead it feeds the weighted composite and can disqualify pairs via the composite threshold):
- **Correlation stability** uses **60** trailing trading days versus a **252**-day baseline. If recent Pearson correlation falls below **0.50**, its component score collapses toward **zero**, otherwise it scores how stable recent correlation is versus the baseline.
- **Volatility** compares short (**20**) and long (**120**) windows on each ticker, builds volatility ratios (`max/std` style within each horizon), mixes them **60%/40%**, returns **inverse-style** compatibility scores, assigns **zero** when data are too short **or** the short-horizon ratio exceeds **2.5×**, or computations fail.

**Sector scores** (static label map embedded in the fundamentals scorer module, mirroring `data/sector_map.py` today): same label **1.0**, cross label **0.4**, unknown ticker **0.5**.

**Ordering before trading:** finalist rows are **`ticker_a` / `ticker_b`** re-ordered whenever the halflife module flags a reversed dependence so **`ticker_a`** is always the dependent leg implied by formation OLS (**log A ~ β log B**). Signals and spreadsheets inherit that canonical ordering.

Formation statistics are recomputed via OLS over the last **120** aligned log-price rows (requires at least roughly **30** overlapping days inside that window — otherwise finalists without β/µ/σ are dropped). The spread volatility fed into z-scoring uses NumPy’s default standard deviation on that spread sample (population-style divide-by-N), matching the scorer implementation.

**Finalists:** within each cluster, sort by composite score and keep the **top 1** pair — so **at most one tradable pair per cluster** after each refresh. If clustering returns no surviving finalists at a refresh, the active finalist map is cleared until a later successful refresh (open positions inherited from older finalists can still trade because exits are processed for anything already open).

---

## Formation statistics — what we lock in for trading

Before live trading decisions, each finalist gets **formation** estimates from **120** trading days of aligned log prices: a **formation β**, **formation mean** of the spread, and **formation standard deviation** of the spread. Those three numbers are fixed until the next rescoring refresh; they define what “normal” and “stretched” mean for signals.

---

## Signals — when we enter and exit

**Today’s spread** (using adjusted closes) is:

spread today = log(price A) − (formation β) × log(price B).

**Formation z-score** is:

z = (spread today − formation mean) ÷ max(formation standard deviation, **0.0001**).

The small floor **0.0001** avoids dividing by a near-zero volatility.

**Entries** (only if flat and regime filters allow):

- **Long the spread** (long A, short B in the dollar-neutral construction) when **z ≤ −1.5**.
- **Short the spread** when **z ≥ +1.5**.

**Exits** for an open position:

- **Take profit** for a long spread when **z ≥ −0.5**; for a short spread when **z ≤ +0.5**.
- **Stop loss** for a long spread when **z ≤ −3.5**; for a short spread when **z ≥ +3.5**.
- **Time stop:** close after **50** trading days open (at least twice the maximum allowed half-life of **20** days by design).

The half-life estimate carried from scoring into the engine is **informational** for bookkeeping; the deterministic calendar stop always uses **`time_stop_days`** (not a multiple of the pair’s estimated half-life intraday).

**Momentum filter** (new entries only): over **14** trading days, if either stock’s cumulative return is beyond **±15%**, we do **not** open — we avoid adding risk during a sharp one-sided move.

**Hedge drift:** each day uses a trailing **60**-day **OLS** hedge ratio (**log A ~ slope × log B**). Once open, live β updating **rescales leg B shares** proportionally (**new/old β**) and offsets cash by the incremental share purchases/sales priced at leg B, then stores the refreshed β anchor so trims are not instantaneous.

---

## Regime filters

**VIX (portfolio level):** if today’s VIX is above **28**, no new positions. New entries resume only after VIX has stayed at or below **25** for **5** consecutive trading days **and** none of those days were above **28**. If there is not enough history to check that, new entries stay off.

**Earnings-style calendar (pair level):** around each **March, June, September, December** quarter-end, no new entries on the last **5** trading days of that month nor the first **1** trading day after quarter-end (either leg triggers the block for that pair). This is a calendar rule, not company-specific announcement dates.

---

## Portfolio sizing and drawdown

Simulation starts with **$100,000**. We keep a **10%** cash buffer, so new trades size off about **90%** of net asset value.

Capital for new trades is split **evenly across the number of active finalist pairs** (`len(active_pairs)`), capped so one leg does not exceed **`max_weight_per_pair`**, here **100%** of NAV (with modest finalist counts the equal-split term usually binds first). After a soft drawdown rule, new size can be cut to **half** (see below).

**Overlap rules in the current implementation:** you cannot hold more open pairs than there are finalist slots, and **the same ticker cannot appear in two different open pairs** at once.

The configuration also defines **maximum gross leverage 2.0×**, **maximum 35%** of active pairs from one subsector, and **at most 2** concurrent pairs from the same cluster — these are **declared targets**; the live simulation today primarily enforces the **pair count** and **no duplicate tickers** unless those extra rules are wired in later.

**Drawdown:**

- **Soft:** roughly weekly, compare today’s NAV to a stored baseline and if the drop exceeds **5%**, new entries shrink to **`drawdown_reduce_factor` = 50%**. In code the baseline snapshots **cash**, not NAV—meaning this trigger can diverge slightly from textbook “versus last week’s NAV”; treat this as implementation detail tied to portfolio accounting.
- **Hard:** when drawdown from the **rolling peak** NAV crosses **10%**, new entries halt and trims fire on subsequent days (**`drawdown_trim_days` = 5** sessions) shrinking each surviving position toward **`drawdown_trim_factor` = 25%** of its outstanding size. Trim cash increments use the signed share reductions × close prices (**not** forced positive), modelling partial unwinds mechanically without separate trim commissions.
- **Recovery from hard halt:** when measured drawdown from peak falls back inside **≤ 5%**, the halt flag resets so normal sizing resumes.

Configuration also declares **`drawdown_recovery_days` = 5**, but today’s portfolio state machine never reads it—recovery is keyed only off the **`drawdown_recovery_threshold`** breach clearing.

---

## Costs, borrow, and execution quality

**Per share commission:** **$0.005** on each leg (absolute share counts).

**Slippage:** **5** basis points of notional if 30-day average volume is above **5 million** shares on that leg; otherwise **10** basis points.

**Bid–ask:** **2** basis points if price is above **$30**, else **5** basis points.

**Short borrow:** **2%** per year, charged daily on the short leg’s position value (using **1/252** of the annual rate per trading day).

**Entry quality check:** expected edge on an open (modeled as long-leg dollar allocation times formation spread standard deviation) must be at least **2.0×** the **round-trip** transaction cost estimate; otherwise the open is skipped.

**Partial fill rule:** if the simulated short leg fills below **95%** of the intended size, the long leg is cancelled and the order may be retried another day.

**Execution quirks:** modeled shorts fill first each day opens are attempted and **borrow** applies daily on whichever leg is net short (**2%**/252). Costs run on fills; closes log realized P&L and exit fees separately from marks.

---
## Simulation day loop — how code orders the pipeline

Each trading day:

1. **Weekly cadence bookkeeping** stores a soft-drawdown baseline snapshot.
2. **Universe reminder** pulls the newest point-in-time list from **`universe_history.parquet`** on the prescribed calendar cadence (**21** calendar days).
3. **Quarterly-ish rescoring** rebuilds correlations, clustering, composite scoring — producing fresh finalists + refreshed formation locks. Exceptions keep the quarterly timer armed so the code retries daily; successful passes always stamp the clock even if no pair survives screening. (Clustering pulls from the processed returns history available that day—usually mirroring the screened universe from `scripts/02`—without re-intersecting the live `load_universe` list inside the engine.)
4. **Mark portfolio** NAV, exposures, gross leverage stats for reporting.
5. **Drawdown machine** adjusts entry permission / sizing scalars.
6. **Hard-halt trims** shave positions over their countdown window.
7. **Portfolio VIX regime** gates new exposure.
8. **Retry pending opens** leftover from thin short fills.
9. **Iterate every union(open pairs, finalist slots)** evaluating signals:
   - Exits flatten entire pairs when thresholds fire.
   - Entries require finalist membership, duplicate-ticker bans, sizing room, volatility & calendar passes, modeled execution + profit-vs-cost hurdle.
10. **Accrue borrow**, increment days-held counters, optionally append diagnostic pair MTM snapshots for CSV exports.

Artifacts written by the scripted backtest commonly include **`trade_log.csv`** (fills + exits), **`nav_series.csv`**, **`pair_daily_mtm.csv`**, and derived analytics spreadsheets under `outputs/`.

---
## End-to-end flow (short)

Hydrate curated **universe** history → periodically **cluster** correlated names → score **within-cluster** pairs → **half-life + Johansen gates** plus composite threshold → **canonicalize** ticker order → lock **formation** β, µ, σ from **OLS** spreads → simulate daily **signals**, **risk**, modeled **fills/costs/borrow**, optional **rescoring resets**.

That is the decision chain plus the bookkeeping layers that sit around it in `src/backtest/` and `scripts/`.
