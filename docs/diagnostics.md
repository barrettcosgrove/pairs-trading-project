# Strategy Diagnostics

Living log of backtest issues, attempted fixes, and measured performance.
Update this file after each material change and re-run of scripts 03–04.

**Window:** 2022-01-03 → 2024-12-31 (`CONFIG.backtest_start_date` / `backtest_end_date`)
**Capital:** $100,000
**Outputs:** `outputs/backtest_results/{trade_log,nav_series,oos_*}.csv`
**OOS slice:** last 30% of the calendar (`CONFIG.oos_fraction = 0.30`), starts 2024-02-08

If code and this file disagree, trust the latest CSVs and then update this file.

---

## Performance across runs

| | Run 0 (baseline) | Run 1 (stop + warmup + no Johansen gate) | Run 2 (coint floor 0.40 + β > 0) | Run 3 (Round 3 fixes + tuned) |
|---|---|---|---|---|
| Ending NAV | $99,522 | $92,264 | $89,435 | $99,026 |
| Total return | −0.48% | −7.7% | −10.6% | −0.97% |
| Annualized | −0.16% | −2.7% | −3.7% | −0.33% |
| Sharpe | −0.18 | −0.83 | −0.96 | −0.07 |
| Max drawdown | 1.4% | 10.0% | 10.6% | 5.4% |
| Round trips | 2 | 22 | 14 | 47 |
| Win rate | 50% (1/2) | 50% (11/22) | 43% (6/14) | **81% (38/47)** |
| Days with exposure | 3 / 753 (0.4%) | 223 / 753 (30%) | 154 / 753 (21%) | 324 / 753 (43%) |
| Last fill | 2023-12-14 | 2023-04-26 | 2023-04-26 | 2024-11-01 |
| OOS trades (2024) | 0 | 0 | 0 | 9 (78% win, −$1.8k) |

### Exit mix (Run 3)

| Exit | N | Net P&L | Avg / trade |
|---|---|---|---|
| TAKE_PROFIT | 38 | +$11,495 | +$303 |
| STOP_LOSS | 8 | −$11,752 | −$1,469 |
| TIME_STOP | 1 | −$170 | −$170 |

Gross spread P&L is roughly symmetric (+$11.5k wins vs −$11.9k losses);
transaction costs (~$2.9k round-trip total) push the net negative. The
binding problem is no longer mechanics — it is per-trade edge: the average
win ($303) is ~4.4× smaller than the average stop (−$1,469), so the 81%
win rate only reaches breakeven.

### Exit mix (Run 2, for reference)

| Exit | N | Net P&L | Avg / trade |
|---|---|---|---|
| STOP_LOSS | 8 | −$12,911 | −$1,614 |
| TAKE_PROFIT | 5 | +$4,850 | +$970 |
| TIME_STOP | 1 | −$2,018 | −$2,018 |

---

## Current issues

### Open

| ID | Issue | Why it matters | Notes |
|---|---|---|---|
| I1b | Per-trade edge ≈ 0 after costs | 81% win rate but avg win $303 vs avg stop −$1,469; net −1% | Gross P&L symmetric; ~$2.9k costs decide the sign. 60+ configs tested — no threshold/sizing/cap combination flipped it in this window |
| I6 | Script 05 / reporting is a stub | No generated chart pack | `scripts/05_generate_report.py` and `src/metrics/reporting.py` are comments only |

### Resolved in Round 3 (do not regress)

| ID | Issue | Fix |
|---|---|---|
| I2 | No entries after spring 2023 | Root cause was the drawdown-halt deadlock (R8), not the entry filters. 2024 now trades (10 exits) |
| I3 | Time-stops do not revert | Tighter entry (1.25) + wider TP (1.0) exits most trades in ~2–3 weeks; 1 time-stop in Run 3 (−$170) |
| I4 | Concentrated pair losses | `max_weight_per_pair` 0.25 cap; no pair can take >25% NAV per leg. Note: further concentration (`target_concurrent_pairs` 4) tested WORSE |
| I5 | β-rebalance churn / sign flips | `rebalance_beta_intra_trade = False`; formation hedge held to exit |

### Resolved (do not regress)

| ID | Issue | Fix |
|---|---|---|
| R1 | Fake stop-loss: z used new β with old μ/σ | Lock formation β/μ/σ for `get_signal`. Rebalance updates `Position.beta_hedge` only (`engine.py`, `portfolio.py`) |
| R2 | Warmup history discarded | Load prices/returns/VIX from 2019; loop only 2022–2024 |
| R3 | Johansen + BH wiped ~99% of pairs | Cointegration is no longer a BH p < 0.10 kill switch. Continuous score `1 − p_adj` |
| R4 | Johansen p = 1.0 if stat < 10% CV | Interpolate p from 1.0 → 0.10 as stat goes 0 → cv10 |
| R5 | Empty rescore cleared `active_pairs` | Keep last quarter’s pairs when scoring returns empty |
| R6 | Johansen only saw 120 days | `CONFIG.johansen_window = 252` |
| R7 | Negative formation β (both legs same direction) | Drop β ≤ `CONFIG.min_formation_beta` (0.0) in `composite.py`; skip at entry |
| R8 | Drawdown halt deadlock — release rule (NAV within 5% of all-time peak) unreachable with entries blocked; froze the book 2023-03-09 → end | Halt is now a cooldown: after trim completes, `peak_nav` resets to current NAV, release after `drawdown_recovery_days` non-losing days (`portfolio.py`) |
| R9 | Trim compounding — 0.25 applied per day for 5 days ≈ liquidation at the low | Per-day step `0.25 ** (1/5)`; position reaches 25% after the trim window (`engine.py::_execute_trim`) |
| R10 | Entry at formation extremes — pair already ≥ entry z on its first active day bought mid-divergence | `entry_requires_cross = True`: enter only the day z first crosses the band (`entry_exit.py`) |
| R11 | Non-deterministic pair processing order (set iteration) could reorder capacity/shared-ticker priority | `sorted()` in the engine loop |

---

## Attempted fixes

### Round 1 — make the pipeline trade honestly

**Date:** 2026-08-31
**Intent:** Fix the false stop, give clustering real history, stop starving the book.

| Change | Where |
|---|---|
| Lock formation β/μ/σ for z-score; `beta_hedge` for share resize | `src/backtest/engine.py`, `src/backtest/portfolio.py` |
| Load all available history; simulate only backtest dates | `src/backtest/engine.py` |
| Johansen is a score, not a hard gate; 252-day window; no p=1.0 cliff | `src/scoring/cointegration.py`, `src/config.py` |
| Half-life remains a hard gate | `src/scoring/composite.py` |
| Keep prior pairs if a cluster date has no finalists | `src/backtest/engine.py` |

**Outcome:** Trades 2 → 22. Days in market 0.4% → 30%. Stops became real (median hold ~24 days). NAV fell $99.5k → $92.3k. Negative-β pair ETN/HON traded. 2024 still idle.

### Round 2 — raise pair quality

**Date:** 2026-08-31
**Intent:** Drop junk that Johansen-as-score-only let through.

| Change | Where |
|---|---|
| Soft floor `min_cointegration_score = 0.40` | `src/config.py`, `src/scoring/composite.py` |
| Reject formation β ≤ 0; pick next-best pair in cluster if top has β ≤ 0 | `src/config.py`, `src/scoring/composite.py` |
| Engine skips entry if β ≤ 0 | `src/backtest/engine.py` |

**Outcome:** Finalists 4–6 → often 2–4. No negative-β trades. Trades 22 → 14. NAV $92.3k → $89.4k. Filtered out Run 1’s best trade (VLO/SLB). Still no 2024 entries. Soft floor is binding (often 100–190 pairs fail `cointegration_score < 0.40`).

---

### Round 3 — fix the deadlock, reshape the payoff, test the levers

**Date:** 2026-08-31
**Intent:** Unfreeze the book (R8/R9), stop buying divergences (R10), then
tune thresholds on the in-sample slice only (IS = dates before 2024-02-08).

**Bug fixes (no tuning):** halt-as-cooldown, trim math, β-rebalance off,
`max_weight_per_pair` 0.25, `finalists_per_cluster` 2, blocked-entry
instrumentation (`blocked_entries.csv`), fresh-cross entry.

**Control run** (fixes only, old 1.5/3.5/0.5 thresholds): NAV $96.3k,
win 58%, 65 round trips, 2024 trades again. Confirmed R8 was the I2 root
cause — post-fix blocks were no_cross 1,223 / momentum 124 / vix 18; the
filters were never the drought.

**Threshold sweeps (IS-only selection, ~60 configs):**

| Direction | Result |
|---|---|
| Wider entry (1.75–2.25), stops 2.75–3.25, TP 0.25–0.75 | Worse. Win 24–52%. Wide entries catch real divergences |
| Tighter entry (1.25–1.5), TP 0.75–1.0, stop 3.5–4.0 | Better. Best: **1.25 / 3.5 / 1.0 → 80% IS win** |
| Tighter stops (2.0–3.0) at entry 1.0–1.25 | No improvement over 3.5 |
| `time_stop_days` 30 | No improvement over 50 |

**Sizing / risk levers (tested at the champion thresholds):**

| Lever | Result |
|---|---|
| `target_concurrent_pairs` 4 (concentrate capital) | WORSE — scales the loss tail faster than wins; dd past halt |
| `max_pair_loss_pct` 0.02 / 0.03 / 0.05 | ALL worse than None — 2–5% dips usually revert; caps realize them |
| `min_cointegration_score` 0.5 → 0.7 (finalists 2) | Monotone better: NAV $94.6k → $99.3k, win 74% → 81%, stops 14 → 8 |
| `min_cointegration_score` 0.85 | Pair supply collapses (23 trades); 0.7 is the knee |
| `finalists_per_cluster` 3 at floor 0.7 | Worse (−$7.2k) — the 3rd-best pair per cluster is junk |

**Final config:** entry 1.25, TP 1.0, stop 3.5, cross-entry on, coint floor
0.70, finalists 2, max weight 0.25, β-rebalance off, no loss cap.

**Outcome (Run 3):** NAV $99,026 (−0.97%), 81% win rate, 47 round trips,
max dd 5.4%, Sharpe −0.07, OOS 9 trades at 78% win (−$1.8k). Mechanics are
fixed; costs eat the remaining ~zero-edge signal (I1b).

**Caveats:** entry/TP/stop chosen on IS data, but the cointegration floor
and lever tests were read on full-period results — treat the −0.97% as
partially in-sample. The honest OOS evidence is 9 trades, 78% win, −$1.8k.

---

## What we believe is *not* the problem

- **Correlation of the universe.** Clustering still produces ~200–250 within-cluster candidates. The drought in Run 0 was the Johansen hard gate, not “stocks don’t move together.”
- **Missing raw data.** `data/raw/prices.parquet` goes back to 2019. Run 0 simply did not load it into the engine.
- **The 3.5 stop threshold itself (Run 0).** The Nov 2023 AVGO/AMAT `STOP_LOSS` fired at true formation z ≈ 1.58 after β was rewritten. Thresholds were not the bug.

---

## Suggested next levers (not done)

These are hypotheses, not commitments. Log any attempt as a new round above.
Levers 1–5 from the previous list were all tested in Round 3 (see above);
the ones that survived are in config, the rest are documented as worse.

1. **Edge, not mechanics** — the remaining problem is per-trade expectancy
   (avg win $303 vs avg stop −$1,469, costs ~$2.9k). Candidates: signal
   refinements outside the current z-score family (e.g. spread-vol scaling,
   regime-conditional entries), or a different holding-period target.
2. **Cash yield accounting** — a real dollar-neutral book earns T-bill rate
   on ~full NAV (2–5% in this window, ≈ +$12k). Deliberately NOT credited:
   headline NAV would clear $110k while trading alpha stayed ~zero.
   Decision was to keep trading-only P&L.
3. **Reporting** — implement `scripts/05_generate_report.py` /
   `src/metrics/reporting.py`, including the blocked_entries breakdown.

---

## How to re-run

Fixes above do not change universe filters. Do **not** re-fetch (01) or rebuild the universe (02) unless `src/universe/` or filter config changes.

```bash
uv run python scripts/03_run_backtest.py
uv run python scripts/04_walkforward.py
```

Compare `outputs/backtest_results/nav_series.csv` and `trade_log.csv`, then add a row to the performance table and a short “Round N” section.
