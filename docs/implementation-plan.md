# Implementation Plan

# ARQ Pairs Trading — Implementation Plan

This is the team's day-to-day coordination document. It is updated at
standup each morning — mark tasks complete, add blockers, and update
ownership as work shifts. The goal is that anyone can open this file and
know exactly where the project stands.

**Standup format (15 minutes, daily):**
1. What did you finish yesterday?
2. What are you doing today?
3. What is blocking you?

---

## Team and Module Ownership

| Member | Owns | Presentation Section |
|---|---|---|
| Barrett | `src/data/`, `src/universe/`, `src/config.py`, `data/sector_map.py`, `scripts/01`, `scripts/02`, `tests/fixtures/`, repo setup | Universe selection, data pipeline, known limitations |
| Althan | `src/clustering/`, `src/scoring/` | Clustering methodology, composite score components |
| Anvay | `src/tiering/`, `src/signals/`, `src/regime/`, `src/backtest/` | Signal generation, backtest results, trade simulation |
| Nanshu | `src/metrics/`, `scripts/03`, `scripts/04`, `scripts/05` | Performance metrics, charts, walk-forward results |

---

## Milestones

| Milestone | Date | Pass Condition |
|---|---|---|
| M1 — Data pipeline complete | ✅ Done | `scripts/02_build_universe.py` runs, 65 avg tickers/month, all processed parquets exist |
| M2 — All modules complete | Day 7 | `pytest` passes all test files, no stubs remaining |
| M3 — Integration complete | Day 9 | Backtest runs end-to-end without crashing, trade log exists |
| M4 — Results complete | Day 11 | OOS metrics documented, all report charts generated |
| M5 — Submission | Day 13 | Report submitted, repo tagged |

---

## Agreed Data Contracts

These function signatures are frozen. Do not change without notifying the
affected owner and updating CLAUDE.md.

```python
# Barrett → Everyone
load_returns(start: date, end: date) -> pd.DataFrame
    # columns: [date, ticker, log_return]
    # sorted by date ascending, no NaN values

load_prices(start: date, end: date) -> pd.DataFrame
    # columns: [date, ticker, adj_close, volume]

load_vix(start: date, end: date) -> pd.Series
    # index: date, values: vix close, name: "vix"

load_universe(as_of: date) -> list[str]
    # tickers passing all hard filters on that date

# Althan → Anvay
run_clustering(distance_matrix: pd.DataFrame) -> dict[int, list[str]]
    # keys: cluster id, values: list of tickers

score_candidates(
    clusters: dict[int, list[str]],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: date,
) -> pd.DataFrame
    # columns: [ticker_a, ticker_b, cluster_id, composite_score]
    # top 2-3 per cluster, sorted by score descending

# Anvay → Nanshu
get_signal(ticker_a, ticker_b, prices, as_of) -> str
    # one of: LONG_SPREAD, SHORT_SPREAD, TAKE_PROFIT,
    #         STOP_LOSS, TIME_STOP, HOLD

run_backtest(config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]
    # returns (trade_log, nav_series)
```

---

## Phase 1 — Foundation (Days 1–2)

**Goal:** Everyone has working code to build against. No module
dependencies yet. Data pipeline complete by end of Day 2.

Barrett is the critical path. Nothing else can start until
`data/raw/prices.parquet` exists and `load_returns()` works.

---

### Block 1 — Foundation Setup

Barrett builds the project skeleton and data fetcher. Everyone else reads
docs and plans their module.

| Owner | Task | File(s) | Status | Notes |
|---|---|---|---|---|
| Barrett | Initialize repo, push full file structure with empty stubs | All top-level dirs and stub files | [x] | Repo initialized, structure committed |
| Barrett | Write `pyproject.toml`, `.gitignore`, `CLAUDE.md` | `pyproject.toml`, `.gitignore`, `CLAUDE.md` | [x] | Committed |
| Barrett | Write `src/config.py` with all parameters | `src/config.py` | [x] | All CONFIG fields defined; never hardcode values elsewhere |
| Barrett | Write `src/data/fetch.py` and `scripts/01_fetch_data.py` | `src/data/fetch.py`, `scripts/01_fetch_data.py` | [x] | Batch price download + staged fundamentals/regime with resume; `--disable-proxy` bypasses network blocks |
| Barrett | Run fetch, share `data/raw/` via Google Drive | `data/raw/*.parquet` | [x] | Proxy issue resolved with `--disable-proxy`; `data/raw/` shared on Drive |
| Barrett | Draft `docs/architecture.md`, `docs/data.md` | `docs/architecture.md`, `docs/data.md` | [x] | Committed |
| Althan | Read all docs, sketch function signatures for clustering + scoring | — | [ ] | |
| Anvay | Read all docs, write pseudocode for backtest daily loop | — | [ ] | |
| Nanshu | Read all docs, set up local environment, run `uv sync` | — | [ ] | |

---

### Block 2 — Data Pipeline Completion

Barrett delivers the complete data pipeline. Milestone M1 closes at end of
this block. All other modules depend on Barrett's outputs being in place.

| Owner | Task | File(s) | Status | Notes |
|---|---|---|---|---|
| Barrett | Write `src/data/clean.py` — missing day handling, validation | `src/data/clean.py` | [x] | Forward-fills single-day gaps; drops tickers with >CONFIG.max_missing_days in any trailing CONFIG.data_quality_window |
| Barrett | Write `src/data/load.py` — typed loader functions | `src/data/load.py` | [x] | **Unblocks all other modules** — `load_returns()`, `load_prices()`, `load_vix()`, `load_universe()` |
| Barrett | Write `src/universe/filter.py` — all 5 hard pre-filters | `src/universe/filter.py` | [x] | Sector, liquidity, price, market cap proxy, SPY correlation; writes `universe_history.parquet` |
| Barrett | Write `data/sector_map.py` — all tickers mapped to subsectors | `data/sector_map.py` | [x] | Every CANDIDATE_TICKER must have an entry or filter.py raises KeyError |
| Barrett | Write `scripts/02_build_universe.py` and run it | `scripts/02_build_universe.py` | [x] | Orchestrates clean → returns → universe history |
| Althan | Write `src/clustering/correlation.py` — distance matrix | `src/clustering/correlation.py` | [ ] | |
| Anvay | Write stubs for `src/tiering/assign.py`, `src/signals/spread.py` | `src/tiering/assign.py`, `src/signals/spread.py` | [ ] | Stubs with correct signatures, raise NotImplementedError |
| Nanshu | Write stub for `src/metrics/performance.py` | `src/metrics/performance.py` | [ ] | |
| Everyone | Clone repo, run `uv sync`, confirm environment works | — | [ ] | |

**End of Block 2 checkpoint (M1):**
- [x] `scripts/02_build_universe.py` runs without errors
- [x] `data/processed/returns.parquet` exists
- [x] `data/processed/universe_history.parquet` exists
- [x] All teammates can `from src.data.load import load_returns` without error

**M1 COMPLETE** — `universe_history.parquet` has 42 recon dates, 65 avg passing tickers/month (56–72 range).

---

## Phase 2 — Core Build (Days 3–7)

**Goal:** Every module implemented and passing its own unit tests.
No integration yet — each module is built and tested in isolation.

---

### Day 3

| Owner | Task | Status | Notes |
|---|---|---|---|
| Barrett | Write `tests/fixtures/` — synthetic price matrix, spread, NAV series | [ ] | Unblocks everyone's tests |
| Barrett | Support role — PR reviews, help debug environment issues | [ ] | |
| Althan | Write `src/clustering/kmeans.py` — silhouette-scored K-means | [ ] | |
| Anvay | Write `src/tiering/kpss.py` and `src/tiering/assign.py` | [ ] | |
| Nanshu | Write `src/backtest/portfolio.py` — position tracking, NAV, tier pools | [ ] | |

---

### Day 4

| Owner | Task | Status | Notes |
|---|---|---|---|
| Barrett | Write `tests/test_clustering.py` and `tests/test_backtest.py` | [ ] | |
| Althan | Write `src/scoring/correlation_stability.py` and `src/scoring/halflife.py` | [ ] | |
| Anvay | Write `src/signals/hedge_ratio.py` — rolling OLS, beta flip detection | [ ] | |
| Nanshu | Write skeleton of `src/backtest/engine.py` — daily loop structure | [ ] | No signal logic yet, just the loop |

---

### Day 5

| Owner | Task | Status | Notes |
|---|---|---|---|
| Barrett | Write `tests/test_scoring.py` and `tests/test_signals.py` | [ ] | |
| Althan | Write `src/scoring/cointegration.py` with BH FDR correction | [ ] | |
| Althan | Write `src/scoring/volatility.py` — dual window ratio | [ ] | |
| Althan | Write `src/scoring/fundamentals.py` — P/S and revenue growth | [ ] | |
| Anvay | Write `src/signals/spread.py` and `src/signals/entry_exit.py` | [ ] | |
| Nanshu | Write `src/regime/bollinger.py` and `src/regime/vix.py` | [ ] | |

---

### Day 6

| Owner | Task | Status | Notes |
|---|---|---|---|
| Althan | Write `src/scoring/composite.py` — gates, normalization, ranking | [ ] | |
| Anvay | Write `src/regime/earnings.py` — end-of-quarter blackout | [ ] | |
| Anvay | Complete `src/backtest/engine.py` — wire in filters and signals | [ ] | Not yet connected to real data |
| Nanshu | Write `src/backtest/costs.py` and `src/backtest/execution.py` | [ ] | |
| Everyone | All unit tests for your module passing by end of day | [ ] | |

---

### Day 7 — Buffer and Cleanup

This day is intentionally light on new tasks. Use it to finish anything
that slipped from Days 3–6, fix failing tests, improve docstrings, and
review each other's PRs.

| Owner | Task | Status | Notes |
|---|---|---|---|
| Everyone | Finish any incomplete module tasks from Days 3–6 | [ ] | |
| Everyone | All unit tests passing (`pytest` runs clean) | [ ] | |
| Everyone | All PRs from Phase 2 merged into develop | [ ] | |
| Barrett | Review sector_map.py — confirm every CANDIDATE_TICKER has a subsector | [ ] | |

**End of Day 7 checkpoint (M2):**
- [ ] Every module is implemented — no `NotImplementedError` remaining
- [ ] `uv run pytest` passes all four test files
- [ ] All Phase 2 PRs merged into develop

---

## Phase 3 — Integration (Days 8–9)

**Goal:** The full pipeline runs end-to-end on real data. Results may be
wrong — that is expected. Getting it to run without crashing is the goal.

Anvay drives integration since they own the backtest engine. Everyone else
is on call to fix issues in their own module as they surface.

---

### Day 8 — Wire It Together

| Owner | Task | Status | Notes |
|---|---|---|---|
| Anvay | Write `scripts/03_run_backtest.py`, run it, fix crashes | [ ] | It will crash. That is normal. |
| Barrett | On call — fix data issues (wrong column names, missing dates, parquet schema mismatches) | [ ] | |
| Althan | On call — fix clustering/scoring issues (NaN scores, empty clusters, wrong matrix shapes) | [ ] | |
| Nanshu | On call — fix metrics/reporting issues | [ ] | |

**Common integration bugs to pre-diagnose:**

| Bug | Likely Cause | Fix |
|---|---|---|
| `KeyError` on column names | Inconsistent column naming between modules | Standardize in `load.py` |
| NaN spread values | OLS on window shorter than `signal_window` | Add minimum window guard in `hedge_ratio.py` |
| Empty tier pools | Johansen + BH correction eliminating all pairs | Temporarily lower `johansen_threshold` to 0.15 to diagnose |
| NAV goes negative | Cost model sign error or missing dollar-neutral enforcement | Check `costs.py` sign conventions |
| Zero trades executed | Regime filters blocking everything | Check VIX levels in the backtest period |

---

### Day 9 — Integration Debug + First Results

| Owner | Task | Status | Notes |
|---|---|---|---|
| Anvay | Fix remaining integration errors, get backtest to complete a full run | [ ] | |
| Barrett | Audit data outputs — spot-check that `load_returns()` matches raw prices | [ ] | |
| Althan | Audit composite scores — verify weights sum to 1.0, normalization is per-cluster | [ ] | |
| Nanshu | Write `src/metrics/performance.py`, compute first Sharpe from trade log | [ ] | |

**End of Day 9 checkpoint (M3):**
- [ ] `uv run python scripts/03_run_backtest.py` completes without crashing
- [ ] `outputs/backtest_results/trade_log.csv` exists
- [ ] `outputs/backtest_results/nav_series.csv` exists
- [ ] At least 5 pairs traded at some point during the backtest

---

## Phase 4 — Validation and Results (Days 10–11)

**Goal:** Results you can defend. OOS period validated. Walk-forward run.

---

### Day 10

| Owner | Task | Status | Notes |
|---|---|---|---|
| Barrett | Write `src/metrics/reporting.py` — NAV curve, drawdown, exit breakdown | [ ] | |
| Althan | Run sensitivity analysis — vary `entry_percentile` (1/2/5), `signal_window` (45/60/90) | [ ] | |
| Anvay | Run OOS period (held-out 6 months) — compare metrics to in-sample | [ ] | Flag any metric differing by >30% |
| Nanshu | Write `scripts/04_walkforward.py` — quarterly rolling windows | [ ] | |

---

### Day 11 — Results Analysis

| Owner | Task | Status | Notes |
|---|---|---|---|
| Everyone | Analyze results together — what drove performance, which pairs contributed, which exit type dominated | [ ] | |
| Nanshu | Run `scripts/05_generate_report.py` — produce all final charts | [ ] | |
| Anvay | Write performance summary — tables, metrics, benchmark comparison vs XLK | [ ] | |

**End of Day 11 checkpoint (M4):**
- [ ] All charts generated in `outputs/report/`
- [ ] OOS results documented
- [ ] Walk-forward results documented (even if partial)
- [ ] Team has a clear explanation of why the strategy performed as it did

**Hard rule: No new code after end of Day 11.**
Days 12–13 are report only. If something is broken, document it honestly
rather than scrambling to fix it at the cost of the writeup.

---

## Phase 5 — Report and Submission (Days 12–13)

**Goal:** Clean, well-written deliverable ready for submission.

---

### Day 12 — Report Draft

| Owner | Writes | Status |
|---|---|---|
| Barrett | Data pipeline section — sources, cleaning decisions, known limitations (survivorship bias, fundamental staleness) | [ ] |
| Althan | Methodology section — clustering rationale, composite score components, why Johansen over Engle-Granger | [ ] |
| Anvay | Signal generation and results section — empirical percentiles, regime filters, backtest metrics, benchmark comparison | [ ] |
| Nanshu | Walk-forward section, charts integration, report assembly | [ ] |

---

### Day 13 — Polish and Submit

| Task | Owner | Status |
|---|---|---|
| Full draft review — catch inconsistencies, check all charts referenced | Everyone | [ ] |
| Final edits and formatting | Everyone | [ ] |
| Confirm all outputs regenerate cleanly from scratch | Nanshu | [ ] |
| Tag repo at submission commit | Barrett | [ ] |
| Submit | Everyone | [ ] |

**End of Day 13 checkpoint (M5):**
- [ ] Report submitted
- [ ] Repo tagged at final commit
- [ ] `data/raw/` shared on Google Drive for record-keeping

---

## Scope Cut Decision Tree

If the team falls behind, cut in this order. Document every cut in
`docs/decisions.md` with a rationale. Cuts that are clearly acknowledged
and explained are academically defensible. Cuts that are hidden are not.

| Behind by | Cut This | Never Cut |
|---|---|---|
| 1 day | Earnings blackout filter (`src/regime/earnings.py`) | Transaction costs in backtest |
| 2 days | Walk-forward validation (`scripts/04_walkforward.py`) | OOS held-out period |
| 3 days | Fundamental compatibility component (set weight to 0, redistribute) | Johansen + KPSS tiering |
| 4+ days | Pair tiering system (single equal pool, no high/low split) | Dollar-neutral position sizing |

---

## Blockers Log

Add blockers here as they arise. Remove them when resolved.

| Date | Blocker | Owner | Status |
|---|---|---|---|
| Block 1 | Proxy/rate limit blocking yfinance | Barrett | Resolved |
| Block 1 | Fix: curl_cffi Session(impersonate="chrome") in fetch.py | | |
| Block 1 | yfinance proxy 403 error blocking data fetch | Barrett | **Resolved** — used `--disable-proxy` flag with `curl_cffi` session (`trust_env=False`) |
| Block 1 | yfinance fundamentals rate limiting | Barrett | Mitigated with staged resume, checkpointing, and slower fundamentals pacing |
| Block 2 | Yahoo rate limiting on regime and fundamentals fetch | Barrett | Workaround: synthetic regime data for development (see `docs/data.md` §8); fundamentals resume with `--stage fundamentals --resume` checkpoint |
| Block 2 | Universe filter market cap proxy too aggressive — only 10 tickers/month | Barrett | Resolved — switched to 30-day avg dollar volume ($25M threshold), now 65 avg/month |
| | | | |

---

## Open Questions

Questions that need a team decision before implementation can proceed.
Move resolved questions to `docs/decisions.md`.

| Question | Owner | Target Resolution | Status |
|---|---|---|---|
| Which package manager — uv or poetry? | Barrett | Block 1 | **Resolved** — uv |
| Share data via Google Drive or each person fetches independently? | Barrett | Block 1 | **Resolved** — Share via Google Drive to avoid rate limiting; Barrett to share `data/raw/` folder |
| Starting capital for backtest — $100K or different? | Anvay | Day 3 | [ ] |
| Fractional shares or round lots in execution model? | Anvay | Day 3 | [ ] |
| Walk-forward window size — quarterly or monthly? | Nanshu | Day 7 | [ ] |
| Comparison benchmark — XLK only, or also equal-weight? | Nanshu | Day 7 | [ ] |

---

## Debug Time Budget

Debug time is explicitly allocated, not hoped for. If a phase runs
over, the buffer below absorbs it before cutting scope.

| Phase | Buffer | Where It Lives |
|---|---|---|
| Phase 1 | 4 hours | Day 2 afternoon — Barrett finishes clean pipeline early |
| Phase 2 | Full Day 7 | Lightest scheduled day, absorbs build overruns |
| Phase 3 | All of Day 9 | Dedicated integration debug day |
| Phase 4 | Day 11 afternoon | Results investigation |
| Total | ~3 days | ~23% of the sprint |