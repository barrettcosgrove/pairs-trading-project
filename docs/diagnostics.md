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

| | Run 0 (baseline) | Run 1 (stop + warmup + no Johansen gate) | Run 2 (coint floor 0.40 + β > 0) | Run 3 (Round 3 fixes + tuned) | Run 4 (entry band + plateau) | Run 5 (earnings exit + 0.35 cap, FINAL) |
|---|---|---|---|---|---|---|
| Ending NAV | $99,522 | $92,264 | $89,435 | $99,026 | $99,918 | **$103,276** |
| Total return | −0.48% | −7.7% | −10.6% | −0.97% | −0.08% | **+3.28%** |
| Sharpe | −0.18 | −0.83 | −0.96 | −0.07 | 0.01 | **0.34** |
| Max drawdown | 1.4% | 10.0% | 10.6% | 5.4% | 5.4% | 5.1% |
| Round trips | 2 | 22 | 14 | 47 | 45 | 45 |
| Profitable trades | 50% (1/2) | 50% (11/22) | 43% (6/14) | 81% (38/47) | 82% (37/45) | 76% (34/45) |
| Worst trade | — | — | — | −$4,023 | −$4,059 | −$1,706 |
| Last fill | 2023-12-14 | 2023-04-26 | 2023-04-26 | 2024-11-01 | 2024-11-01 | 2024-10-22 |
| OOS trades (2024) | 0 | 0 | 0 | 9 (78% win, −$1.8k) | 9 (−$1.8k) | **9 (7/9, +$3.9k)** |

### Exit mix (Run 5, final)

| Exit | N | Net P&L | Avg / trade |
|---|---|---|---|
| TAKE_PROFIT | 34 | +$13,411 | +$394 |
| EARNINGS_EXIT | 6 | −$4,121 | −$687 |
| PLATEAU_STOP | 3 | −$3,680 | −$1,227 |
| STOP_LOSS | 1 | −$1,706 | −$1,706 |
| TIME_STOP | 1 | −$166 | −$166 |

Realized trade P&L +$3,737. One hard stop remains in three years (TMO/DHR,
March 2023 — the SVB week; no ex-ante feature flags it). The 76% profitable
rate is lower than Run 4's 82% because six pre-earnings exits realize small
controlled losses (avg −$687) in exchange for removing −$4k single-day
earnings-gap tails; the exchange is what turns the OOS slice from −$1.8k
to +$3.9k.

### Exit mix (Run 3, for reference)

| Exit | N | Net P&L | Avg / trade |
|---|---|---|---|
| TAKE_PROFIT | 38 | +$11,495 | +$303 |
| STOP_LOSS | 8 | −$11,752 | −$1,469 |
| TIME_STOP | 1 | −$170 | −$170 |

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
| I6 | Script 05 / reporting is a stub | No generated chart pack | `scripts/05_generate_report.py` and `src/metrics/reporting.py` are comments only |
| I7 | Round 4/5 thresholds partially selected on full-period data | +3.28% is partly in-sample | Entry band 2.0 and plateau 2.75/3 were chosen from Run 3 trade forensics (mostly IS trades); `earnings_exit_min_adverse_z` 1.75 was picked from a 2-point test (1.5 vs 1.75) that includes the OOS trades. SCHW/MS exits at adverse 1.89 — a threshold above that gives the −$4k gap back. Treat 1.75 as fragile until more earnings events are observed |
| I8 | Win rate 76% vs 82% without earnings exits | User-facing metric regression | Six EARNINGS_EXIT scratches (avg −$687) count as losses; setting `earnings_exit_days_before = 0` restores 82% profitable but returns the −$4k tail and OOS −$1.8k. Deliberate trade |

### Resolved in Round 4/5 (do not regress)

| ID | Issue | Fix |
|---|---|---|
| I1b | Per-trade edge ≈ 0 after costs (Run 3: avg win $303 vs avg stop −$1,469) | Loss tail cut on three fronts: entry band cap (no gap entries), plateau stop (slow bleeders exit ~1σ earlier), conditional pre-earnings exit (gap risk closed while losing). Realized P&L +$3,737, avg win $394, worst trade −$1.7k (was −$4.0k) |

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

### Round 4 — cut the loss tail: entry band cap + plateau stop

**Date:** 2026-08-31
**Intent:** Attack I1b from the loss side. New trade-log instrumentation
(z_entry/z_exit, formation stats, 20-day spread vol ratio on every trade)
plus per-trade z-path reconstruction showed the 8 Run 3 stops were not one
phenomenon but three:

1. **Gap entries.** The fresh-cross rule accepted crosses that jumped from
   inside the band to far beyond it in one day: EXC/SO entered at z=−4.91
   (beyond the 3.5 stop itself, vol_ratio 0.99) and stopped the next day at
   −8.43 (−$923); GS/MS entered at 2.22 and stopped in 3 days (−$358).
2. **Slow bleeders.** TMO/DHR, NOW/ANET, WEC/XEL, SO/WEC lingered at
   adverse z 2.4–3.3 for 1–4 weeks without ever re-approaching the mean,
   then breached 3.5. Reversion had failed long before the stop fired.
3. **Earnings gaps** (deferred to Round 5): SCHW/MS −$4.0k and CVX/VLO
   −$1.6k showed *no* entry-time warning; both were one-day earnings gaps
   through the stop (z_exit −5.65 and +3.53).

Stops also gap *through* the 3.5 line (realized exit z 3.9–8.4), so the
loss per stop is far worse than the theoretical 2.25σ. MAE analysis showed
winners endure little pain (median −0.3% of allocation) but the stop-day
gap delivers most of each loss — which is also why the Round 3 dollar-cap
tests failed: caps can't dodge gaps but do clip slow-grinding winners.

| Change | Where |
|---|---|
| `entry_zscore_max = 2.0` — entries only inside the [1.25, 2.0] band | `src/config.py`, `src/signals/entry_exit.py` |
| Plateau stop: adverse z ≥ 2.75 for 3 consecutive days → PLATEAU_STOP (STOP_LOSS cooldown applies) | `src/config.py`, `src/signals/entry_exit.py`, `src/backtest/engine.py` |
| Blocked-entry reason `entry_band`; trade-log diagnostic columns | `src/backtest/engine.py` |

A plain tighter stop had tested worse in Round 3 because single-day spikes
to ~2.8 usually revert; the 3-day persistence requirement is what makes
the earlier exit safe (what-if: P=2.75/D=3 clipped zero winners).

**Outcome (Run 4):** NAV $99,918 (−0.08%), 45 round trips, 82% profitable,
realized trade P&L +$420 (first positive), max dd 5.4%. Exit mix:
TAKE_PROFIT 37 / +$11.4k, STOP_LOSS 3 / −$6.9k (TMO/DHR SVB-week, SCHW/MS,
CVX/VLO), PLATEAU_STOP 4 / −$3.9k, TIME_STOP 1. Freed capacity admitted one
new loser (TXN/QCOM Sep-23 plateau −$1.1k). OOS still −$1.8k — entirely
the two earnings gaps.

### Round 5 — defensive pre-earnings exit on real earnings dates

**Date:** 2026-08-31
**Intent:** Kill the earnings-gap tail (the whole remaining OOS loss)
without giving back the wins that near-earnings entries produce.

New data + plumbing: `data/raw/earnings.parquet` (real per-ticker report
dates via a new `earnings` fetch stage in `fetch.py` / scripts 01;
`load_earnings_dates()` in `load.py`; `earnings_within()` in
`regime/earnings.py`; requires `lxml`). Coverage: all 95 tickers,
2007–2026.

Rule: close an open pair `earnings_exit_days_before` (2) trading days
before either leg reports **only if** adverse formation z ≥
`earnings_exit_min_adverse_z`; no cooldown afterwards (the pair may
re-enter on a fresh cross once the event passes). Winning/flat positions
are held through earnings — DE/HON +$963, ADSK/WDAY +$775, TXN/QCOM +$621
were all entered within 5 trading days of a report and won, so an
unconditional exit or entry blackout gives the edge back. The quarter-end
`in_blackout` proxy (which almost never coincides with real report dates)
is left untouched.

**Threshold and sizing tests (full engine runs, weight cap 0.25 unless noted):**

| Variant | NAV | Profitable | OOS net | Notes |
|---|---|---|---|---|
| adverse ≥ 1.50 | $100,552 | 33/48 (69%) | +$1.4k | 11 earnings exits; clipped 7 marginal positions (GS/MS at 1.59, DE/HON at 1.51…) that mostly recovered |
| adverse ≥ 1.75 | $102,389 | 34/45 (76%) | +$2.8k | 6 exits; keeps the four real dangers (SCHW 1.89, CVX 1.80, TMO 2.61, ADSK/CRM 2.17), holds the marginal ones |
| adverse ≥ 1.75 + `max_weight_per_pair` 0.35 | **$103,276** | 34/45 (76%) | +$3.9k | Sharpe 0.33 → 0.34, max dd 4.2% → 5.1% — pure linear scaling of a now-positive expectancy; adopted |

**Outcome (Run 5, final config):** NAV $103,276 (+3.28%), Sharpe 0.34,
max dd 5.1%, 45 round trips, 76% profitable, realized +$3,737, OOS slice
(2024-02-08 →) +4.79% NAV with 7/9 profitable trades. DE/HON was held
through its 2024-05-16 report (adverse < 1.75) and took profit +$942 the
same week — the conditionality works in both directions.

**Caveats:** see I7 — the adverse-z threshold sits 0.14 below SCHW/MS's
exit reading, and part of the selection evidence is the OOS window itself.
The structural claims (gap entries are bad, plateaus don't revert, losing
positions into earnings carry open-ended gap risk) are mechanism-backed;
the exact numbers are tuned.

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
