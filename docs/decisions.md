# Architecture Decision Log

<!-- Records every significant design decision, why it was made, and what alternatives were rejected. Prevents revisiting settled decisions. -->

## Fundamental compatibility weight temporarily set to 0.0

Date: April 2026

Decision: Set `weight_fundamentals = 0.0` in `config.py` temporarily.

Rationale: yfinance fundamentals endpoint rate-limited on development
network. Will restore to 0.10 once `fundamentals.parquet` is fetched
on hotspot. Other weights redistributed proportionally.

Restore procedure: fetch fundamentals on hotspot, update `config.py`
weights back to original values, rebuild processed data.

## Universe filter: market cap proxy replaced with dollar volume threshold

Date: April 2026

Decision: Renamed `min_market_cap` to `min_dollar_volume` in `config.py`
and set threshold to $25,000,000 (30-day average daily dollar volume).

Problem: The original filter used single-day `adj_close × volume` compared
against a $2B threshold. This is a category error — `adj_close × volume`
is daily dollar trading volume, not market cap. A $2B daily dollar volume
threshold is extremely high and eliminated all but 10 mega-cap stocks per
month, leaving too few candidates for clustering.

Alternatives rejected:
- Fetch actual market cap (shares outstanding not in OHLCV — would require
  separate API call per ticker)
- Keep $2B threshold with single-day volume (too volatile, wrong metric)

Fix: 30-day average daily dollar volume at $25M threshold. This correctly
filters micro-caps and thinly traded names while keeping mid and large cap
tech stocks. Universe now averages 65 passing tickers per month (56–72
range across 42 reconstitution dates).

## Kmeans Clustering: Change stock universe from tech stock to sectors in S&P500

Decision: use all sectors from S&P500 instead of only tech stocks

Problem: When running k-means on samples, Silhouette scoring gets values that are typically 
less than 0.15, which suggests that they are weak and no better than random. This is due
to the fact that we are only using tech stocks, where they are all driven by NASDAQ, and
that tech stocks move together 80-90% of the time.

Fix: Replace our universe so that it contains more diversity with more sectors.
This can include stocks in sectors like finanicals, energy, consumer_cyclical, healthcare,
industrials, and also tech.

## Drawdown halt is a temporary circuit breaker, not a permanent stop

Date: August 2026 (Round 3)

Decision: After the 10% hard halt trims positions, the rolling peak NAV is
reset to the post-trim NAV and the halt releases after
`drawdown_recovery_days` (5) consecutive non-losing days. The trim reduces
positions to `drawdown_trim_factor` (25%) of pre-halt size over
`drawdown_trim_days` — a per-day multiplicative step, not the full factor
each day.

Problem: The old release rule required NAV within 5% of the *all-time* peak
while entries were blocked and positions were trimmed toward zero
(0.25^5 ≈ 0.1% of size — a forced liquidation at the drawdown low). NAV
could never recover, so the halt was a deadlock: it fired 2023-03-09 and
blocked every entry for the remaining 21 months, freezing NAV at $89,435
and leaving the 2024 OOS period with zero trades.

Alternatives rejected:
- Disable the halt entirely (loses the circuit breaker in a 2022-style
  selloff)
- Raise the threshold to 15% (deep drawdown could still deadlock)

## Intra-trade beta rebalancing disabled

Date: August 2026 (Round 3)

Decision: `rebalance_beta_intra_trade = False`. The formation hedge ratio
is held from entry to exit. The rebalancing code path remains behind the
flag.

Problem: The 0.15 *absolute* deadband on a noisy 60-day rolling beta caused
frequent resizes of leg B that realized cash buy-high/sell-low, and scaling
`shares_b` by `new_beta / old_beta` could flip the short leg's sign when
live beta crossed zero (UNH/ABBV), turning the hedge into a directional
bet. One stop-loss even booked positive P&L because rebalance cash
adjustments dominated the spread P&L.

## Entries require a fresh z-score cross

Date: August 2026 (Round 3)

Decision: `entry_requires_cross = True`. A flat pair enters only on the
first day |z| moves beyond the entry band (prior day inside it).

Problem: On each quarterly score date, z is measured against the pair's own
formation window, so any pair already stretched past the entry threshold
fired immediately at the close — buying into an actively diverging spread.
SCHW/BAC entered on day 2 of the backtest and stopped out in 8 days, twice.

## Per-pair concentration cap and doubled finalists

Date: August 2026 (Round 3)

Decision: `max_weight_per_pair` 1.0 → 0.25 (per-leg allocation ≤ 25% of
NAV); `finalists_per_cluster` 1 → 2.

Problem: Equal-weight sizing over the active-pair count let a single pair
take ~90% of NAV when few pairs were active (WEC/XEL received an $85k
allocation on $100k NAV). With only 4–6 active pairs the book was too
concentrated and too idle; a handful of stops dominated the P&L.

## Cointegration floor raised to 0.70; sizing levers rejected by data

Date: August 2026 (Round 3)

Decision: `min_cointegration_score` 0.40 → 0.70. Kept
`target_concurrent_pairs = 10` (equal split across active pairs) and
`max_pair_loss_pct = None` after testing the alternatives.

Evidence: The floor curve was monotone-better 0.5 → 0.7 (NAV $94.6k →
$99.3k, win 74% → 81%, stop-losses 14 → 8) and collapsed pair supply at
0.85. Concentrating capital (`target_concurrent_pairs` 4) and per-pair
dollar caps (2–5%) each LOWERED both NAV and win rate — dips of 2–5%
usually revert, and concentration scales the loss tail faster than the
wins. See docs/diagnostics.md Round 3.

## No cash-interest credit on idle NAV

Date: August 2026 (Round 3)

Decision: The backtest credits zero interest on cash, although a real
dollar-neutral book would earn the T-bill rate (2–5% over 2022–2024,
≈ +$12k). Crediting it would push headline NAV past $110k while trading
alpha stayed ~zero — inflating performance rather than demonstrating edge.
Revisit only if the goal changes from "show trading edge" to "model total
account return", and label the interest component separately if so.

## Entry band cap and plateau stop replace part of the hard-stop loss tail

Date: August 2026 (Round 4)

Decision: Added `entry_zscore_max = 2.0` (no entry when |z| already beyond
2.0 — the cross must land inside the 1.25–2.0 band) and a plateau stop
(`stop_plateau_zscore = 2.75`, `stop_plateau_days = 3`: exit after three
consecutive days at or beyond 2.75 adverse). New exit action PLATEAU_STOP;
it triggers the same cooldown as STOP_LOSS.

Evidence: Trade-level instrumentation (z_entry, z_exit, vol_ratio columns
in trade_log.csv) showed (a) EXC/SO "crossed" from inside the band to
z=−4.9 in one day — past the stop itself — and lost −$923 the next day;
GS/MS gapped to 2.2 and stopped in 3 days; (b) every slow-bleed stop-out
(TMO/DHR, NOW/ANET, WEC/XEL, SO/WEC) lingered at z 2.4–3.3 for 1–4 weeks
without re-approaching the mean before breaching 3.5 — exiting on a
sustained plateau realizes ~1σ less loss. A plain tighter stop (2.0–3.0)
had already tested worse in Round 3 because single-day spikes revert; the
persistence requirement is what makes the earlier exit safe. See
docs/diagnostics.md Round 4.

## Defensive pre-earnings exit on real earnings dates; no earnings entry block

Date: August 2026 (Round 5)

Decision: Fetch real per-ticker earnings dates
(`data/raw/earnings.parquet`, new fetch stage). Force-close an open pair
2 trading days before either leg reports, but only when the position is
already losing (adverse z ≥ 1.75; 1.5 tested — it clipped seven marginal
positions that mostly recovered, dropping the profitable rate to 69%).
No cooldown afterwards — the pair may re-enter on a fresh cross once the
event has passed. Deliberately did NOT add an entry blackout around real
earnings dates, and left the quarter-end `in_blackout` approximation
untouched. Also raised `max_weight_per_pair` 0.25 → 0.35: with the loss
tail cut, expectancy is positive and scales linearly (Sharpe 0.33 → 0.34,
max dd 4.2% → 5.1%); do not raise further without retesting concentration.

Evidence: The two largest losses in the book were single-day earnings gaps
through the z-stop — SCHW/MS −$4.0k (SCHW −10% on its 2024-07-16 print,
z −2.3 → −5.6 in one day) and CVX/VLO −$1.6k (VLO 2024-10-24 / CVX
2024-11-01) — both in the OOS window; no entry-time feature (vol ratio,
entry z) flagged them. Losing positions ahead of a print carry open-ended
gap risk, while entries near earnings repeatedly WON (DE/HON +$963,
ADSK/WDAY +$775, TXN/QCOM +$621 were all opened within 5 trading days of
a report): an unconditional exit or entry blackout gives back ~$4k of
wins. Conditional exit what-if: net +$3.1k vs +$0.4k, OOS −$1.9k → +$2.8k.
See docs/diagnostics.md Round 5.
