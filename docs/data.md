# Data Reference

This document is the reference for fetch sources, parquet schemas, cleaning,
and known data issues. Live scoring uses sector labels, not the fundamentals
snapshot. Pipeline layout: [`architecture.md`](architecture.md).

If you are debugging a data issue, start here.

---

## 1. Overview

The candidate universe is ~95 multi-sector S&P-style names in
`CANDIDATE_TICKERS` (`src/data/fetch.py`), not a tech-only GICS 45 list.
Fetch window: 2019-01-01 → 2026-04-01.


| Source                       | What It Provides                        | Fetch Frequency           |
| ---------------------------- | --------------------------------------- | ------------------------- |
| Yahoo Finance (prices)       | Daily OHLCV for all candidate tickers   | Once at project start     |
| Yahoo Finance (fundamentals) | P/S ratio and TTM revenue growth        | Once — **unused by live scoring** |
| CBOE then yfinance (VIX)     | Market volatility index                 | Once at project start     |
| Alpha Vantage then yfinance (SPY) | S&P 500 ETF (needs `ALPHA_VANTAGE_KEY` for AV) | Once at project start |


All raw data is written to `data/raw/` as parquet files immediately after
fetching. Processed data derived from cleaning and filtering is written to
`data/processed/`. Both directories are gitignored — regenerate by running
the fetch and build scripts.

---

## 2. How to Fetch Data

Always use `--disable-proxy` flag which enables Chrome impersonation:

```bash
uv run python scripts/01_fetch_data.py --disable-proxy
```

**Why `--disable-proxy` is required:** Yahoo Finance detects automated
requests by their TLS fingerprint and User-Agent. The `--disable-proxy`
flag creates a curl_cffi session with `impersonate="chrome"` which
mimics a real browser and bypasses bot detection. Without it you will
get HTTP 429 rate limit errors regardless of request frequency.

```bash
# Download everything (run once at project start)
uv run python scripts/01_fetch_data.py --disable-proxy

# Rebuild processed data after fetching or after changing config thresholds
uv run python scripts/02_build_universe.py

# Check for data quality issues after any fetch
cat outputs/data_quality_report.txt
```

If the fetch fails due to a network or proxy issue, see Section 8
(Troubleshooting) before re-running.

`scripts/01_fetch_data.py` runs stages in this order:

1. `prices` — batch OHLCV download for candidate tickers
2. `regime` — VIX and SPY
3. `fundamentals` — P/S ratio and revenue growth snapshot

Regime runs before fundamentals because fundamentals use Yahoo's
quote-summary endpoint, which is the most rate-limit-prone part of the
pipeline.

Useful stage/resume commands:

```bash
# Run only missing outputs; this is the default behavior
uv run python scripts/01_fetch_data.py --disable-proxy --resume

# Resume fundamentals after a rate-limit stop
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals --resume

# Fetch only regime data
uv run python scripts/01_fetch_data.py --disable-proxy --stage regime

# Re-fetch prices but leave regime/fundamentals alone
uv run python scripts/01_fetch_data.py --disable-proxy --stage prices --force

# Run all stages except fundamentals
uv run python scripts/01_fetch_data.py --disable-proxy --skip-fundamentals

# Slow down fundamentals if Yahoo starts rate limiting
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals \
  --fundamentals-delay 10 --rate-limit-cooldown 1800
```

When `--resume` is enabled, the script skips final parquet files that already
exist. Use `--force` when you intentionally want to overwrite an existing raw
file.

### Rate Limiting

Yahoo Finance rate-limits automated requests. If you encounter 429 or
"Too Many Requests" errors during a full fetch, run prices and regime in
separate passes with a wait between them:

```bash
# Step 1 — Fetch prices first (batch call, least rate-limit-prone)
uv run python scripts/01_fetch_data.py --disable-proxy --stage prices

# Step 2 — Wait 5–10 minutes, then fetch regime
uv run python scripts/01_fetch_data.py --disable-proxy --stage regime

# Step 3 — Fetch fundamentals separately (most rate-limit-prone)
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals --resume
```

If rate limiting persists on the regime stage even after waiting, see Section 8
(Troubleshooting) for a script to generate synthetic regime data for development
purposes. Replace synthetic data with a real fetch before the final backtest.

---

## 3. Raw Data Files

These files are written by `src/data/fetch.py` and never modified afterward.
If a file looks wrong, delete it and re-fetch rather than editing it.

---

### 3.1 `data/raw/prices.parquet`

**Written by:** `src/data/fetch.py` → `fetch_prices()`
**Read by:** `src/data/clean.py`, `src/data/load.py`
**Size:** ~500 KB – 2 MB depending on universe size and date range

Daily OHLCV price data for all ~120 candidate tickers over 3.5 years of
history. This is the largest and most important file in the pipeline —
every downstream calculation depends on it.

Fetched using a single `yf.download()` batch call for all tickers rather
than per-ticker requests. The batch approach sends one request to Yahoo
instead of ~120, which significantly reduces rate-limiting exposure.

**Schema:**


| Column      | Type       | Description                               |
| ----------- | ---------- | ----------------------------------------- |
| `ticker`    | str        | Stock ticker symbol (e.g. "AAPL")         |
| `date`      | datetime64 | Trading date                              |
| `open`      | float64    | Opening price in USD                      |
| `high`      | float64    | Intraday high price in USD                |
| `low`       | float64    | Intraday low price in USD                 |
| `close`     | float64    | Raw closing price in USD                  |
| `adj_close` | float64    | Split and dividend adjusted closing price |
| `volume`    | int64      | Number of shares traded                   |


**Primary key:** `(ticker, date)` — one row per ticker per trading day.

**Which price to use where:**

- Use `adj_close` for all return calculations, spread computation,
hedge ratio estimation, and cointegration tests
- Use `close` only where raw price is explicitly needed (e.g. bid-ask
spread tier classification based on price level)
- Use `volume` only for the ADV liquidity filter in universe selection

**Adjustment methodology:**
yfinance applies split and dividend adjustments using its default method.
Adjustments are applied retroactively — historical prices are restated when
a split or dividend occurs. This is standard for backtesting but means the
file should be re-fetched periodically to pick up new adjustments.

---

### 3.2 `data/raw/fundamentals.parquet`

**Written by:** `src/data/fetch.py` → `fetch_fundamentals()`
**Read by:** `src/data/load.py` → `src/scoring/fundamentals.py`
**Size:** ~5–15 KB

Current snapshot of fundamental metrics for each candidate ticker. Fetched
once and held constant over the backtest as a known simplification. See
Section 6.2 for the full discussion of this limitation.

**Schema:**


| Column               | Type    | Description                                                      |
| -------------------- | ------- | ---------------------------------------------------------------- |
| `ticker`             | str     | Stock ticker symbol                                              |
| `fetch_date`         | date    | Date the snapshot was taken (today)                              |
| `price_to_sales`     | float64 | Trailing 12-month price-to-sales ratio                           |
| `revenue_growth_ttm` | float64 | Trailing 12-month revenue growth rate (decimal, e.g. 0.12 = 12%) |


**Missing values:**
Some tickers — particularly smaller or recently listed companies — do not
have P/S or revenue growth available from yfinance. These appear as `NaN`
in the file. The fundamental scorer in `src/scoring/fundamentals.py` treats
missing fundamental data as a neutral score (0.5) rather than discarding
the pair entirely. This is a deliberate choice documented in
`docs/decisions.md`.

**Resume/checkpoint files:**

During the fundamentals stage, partial progress is written to
`data/raw/fundamentals_checkpoint.parquet`. The final
`data/raw/fundamentals.parquet` is written only when the stage completes.
If Yahoo rate-limits the run, wait before retrying and resume with:

```bash
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals --resume
```

The current stopping point is recorded in `data/raw/fetch_status.json`.
Look at `next_ticker`, `last_ticker`, `completed_count`, and `pending_count`
to see where the last run stopped.

---

### 3.3 `data/raw/regime.parquet`

**Written by:** `src/data/fetch.py` → `fetch_regime()`
**Read by:** `src/data/load.py` → `src/regime/vix.py`,
`src/universe/filter.py`
**Size:** ~10–20 KB

Daily VIX and SPY closing prices over the same 3.5-year window as prices.

**Schema:**


| Column | Type       | Description                                          |
| ------ | ---------- | ---------------------------------------------------- |
| `date` | datetime64 | Trading date                                         |
| `vix`  | float64    | CBOE VIX daily close (volatility index, not a price) |
| `spy`  | float64    | SPY adjusted closing price in USD                    |


**How each column is used:**

`vix` — Read by `src/regime/vix.py`. New position entries are blocked when
VIX exceeds `CONFIG.vix_entry_block` (28.0). Entries resume when VIX stays
below `CONFIG.vix_resume` (25.0) for 5 consecutive trading days.

`spy` — Read by `src/universe/filter.py`. The 60-day rolling correlation
between each stock's log returns and SPY log returns is computed. Stocks
with correlation above `CONFIG.max_spy_correlation` (0.90) are excluded
from the universe — they are effectively index proxies with little
company-specific movement to trade.

**Development note:** If Yahoo rate-limits the regime fetch, synthetic VIX
and SPY data can be substituted to unblock development. Use the script in
Section 8 (Troubleshooting) to generate it. Synthetic regime data **must
be replaced with a real fetch before the final backtest** — synthetic data
will have exactly the number of business days in the date range, while
real data may have slightly fewer due to market holidays.

---

### 3.4 `data/raw/earnings.parquet`

**Written by:** `src/data/fetch.py` → `fetch_earnings()`
(`scripts/01_fetch_data.py --stage earnings`)
**Read by:** `src/data/load.py` → `load_earnings_dates()` →
`src/backtest/engine.py` (defensive pre-earnings exit)
**Size:** ~50 KB

Historical and scheduled earnings report dates per candidate ticker,
scraped from Yahoo Finance (`yfinance.Ticker.get_earnings_dates`,
requires the `lxml` package). ~40 quarters per ticker.

**Schema:**

| Column          | Type       | Description                            |
| --------------- | ---------- | -------------------------------------- |
| `ticker`        | object     | Ticker symbol                          |
| `earnings_date` | datetime64 | Report date (tz-naive, normalized)     |

**How it is used:** The backtest engine force-closes an open pair
`CONFIG.earnings_exit_days_before` (2) trading days before either leg
reports, but only when the position is already losing (adverse formation
z ≥ `CONFIG.earnings_exit_min_adverse_z`, 1.5). Earnings dates are public
information announced in advance, so this is look-ahead safe. When the
file is missing the engine logs a warning and disables the feature.

---

## 4. Processed Data Files

These files are generated by `scripts/02_build_universe.py` from the raw
data. Safe to delete and regenerate at any time. If you change filter
thresholds in `src/config.py`, delete `data/processed/` and re-run
`scripts/02_build_universe.py` before running the backtest.

---

### 4.1 `data/processed/returns.parquet`

**Written by:** `src/data/clean.py`
**Read by:** `src/data/load.py` → clustering, scoring, signals

Daily log returns for every ticker that passed the data quality check.
Log returns are used everywhere in the pipeline instead of simple returns
because they are time-additive and make the spread stationary in a
mathematical sense that simple returns often violate.

**Formula:** `log_return = ln(adj_close_t / adj_close_{t-1})`

**Schema:**


| Column       | Type       | Description         |
| ------------ | ---------- | ------------------- |
| `ticker`     | str        | Stock ticker symbol |
| `date`       | datetime64 | Trading date        |
| `log_return` | float64    | Daily log return    |


**Guarantees:**

- No NaN values — rows with missing returns are dropped entirely
- Only tickers that passed the data quality check are included
- Sorted by date ascending within each ticker

---

### 4.2 `data/processed/universe_history.parquet`

**Written by:** `src/universe/filter.py`
**Read by:** `src/data/load.py` → `src/backtest/engine.py`

Records which tickers were in the investable universe at each monthly
reconstitution date. This is what prevents look-ahead bias in universe
construction — the backtest engine queries this file to get the universe
as it would have been on any given historical date, rather than using
today's universe for all historical dates.

**Schema:**


| Column           | Type       | Description                         |
| ---------------- | ---------- | ----------------------------------- |
| `date`           | datetime64 | Monthly reconstitution date         |
| `ticker`         | str        | Stock ticker symbol                 |
| `passed_filters` | bool       | True if all hard pre-filters passed |


**How it is queried:**
`load_universe(as_of=date)` returns the list of passing tickers from the
most recent reconstitution date on or before `as_of`. This ensures the
backtest never uses future universe membership information.

---

### 4.3 `data/processed/correlation_matrices/YYYY-MM.parquet`

**Written by:** `src/clustering/correlation.py`
**Read by:** `src/clustering/kmeans.py`

One parquet file per month containing the full N×N pairwise distance matrix
for that month's 120-day rolling window. Cached to avoid recomputing
correlations on every backtest run.

**Schema:**
NxN DataFrame where both the index and columns are ticker strings. Values
are `1 - correlation`, ranging from 0 (perfectly correlated) to 2
(perfectly anti-correlated).

**Example filename:** `data/processed/correlation_matrices/2023-06.parquet`

---

## 5. Universe Filters

Applied monthly by `src/universe/filter.py`. A stock failing any single
filter is removed from the universe entirely for that month — filters are
binary gates, not scored components.


| Filter          | Threshold                       | Source Field                      | Rationale                                                                                                                                                                                         |
| --------------- | ------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mapped ticker   | Must appear in `SECTOR_MAP`     | `data/sector_map.py`              | Multi-sector labels (Energy, Financials, Healthcare, …). Live fundamentals score uses these labels (same / cross / unknown), not P/S.                                                              |
| Liquidity       | ADV > 1,000,000 shares (30-day) | `volume`                          | Ensures positions can be entered and exited without moving the price                                                                                                                              |
| Price           | Close > $10.00                  | `close`                           | Eliminates penny stocks where bid-ask spreads are wide and price behavior is erratic                                                                                                              |
| Dollar volume   | 30-day avg > $25,000,000        | `adj_close × volume` (30-day avg) | Filters micro-caps and thinly traded names. True market cap unavailable without shares outstanding — 30-day average daily dollar volume is used as a stable size proxy. Documented approximation. |
| SPY correlation | < 0.90 (60-day rolling)         | `adj_close` vs SPY                | Removes stocks that simply track the index and produce spreads driven by macro rather than company-specific dynamics                                                                              |


**Target universe size:** `CONFIG.universe_size` (100) from ~95 mapped candidates across ≥ `CONFIG.min_subsectors` (8) distinct sectors.
**Floor:** If fewer than 60 stocks pass filters in a given month, the
pipeline proceeds with whatever passes and logs a warning.

**Reconstitution timing:** Filters run on the first trading day of each
month. Stocks that fall below thresholds are removed at the month boundary,
not immediately mid-month, to avoid unnecessary churn in active positions.

---

## 6. Known Limitations

These are intentional simplifications made for the project scope. Each
should be discussed in the final report.

---

### 6.1 Survivorship Bias

**What it is:** yfinance only returns data for stocks that are currently
listed. Any stock that was delisted, acquired, or went bankrupt during the
backtest window is entirely absent from the dataset.

**Why it matters:** A real strategy trading in 2022 would have included
some stocks that no longer exist today. Excluding them from the backtest
makes the universe look cleaner and more profitable than it actually was.
Pairs involving a stock that later got acquired would have shown a
cointegration break — a real cost the backtest never sees.

**Mitigation:** The candidate ticker list in `src/data/fetch.py` includes
some names that may have been acquired or delisted (e.g. TWTR). Where
yfinance still has historical data for these, they are included. This
partially reduces the bias but does not eliminate it.

**Impact on results:** Likely overstates Sharpe ratio and win rate by a
small but non-trivial amount. Report this limitation explicitly.

---

### 6.2 Fundamental Data Staleness

**What it is:** P/S ratios and revenue growth figures in
`data/raw/fundamentals.parquet` are fetched as a current snapshot on the
day the fetch script is run. They are not point-in-time historical values.

**Why it matters:** In a backtest spanning 2022–2024, using 2025 P/S ratios
to evaluate pairs in 2022 is technically look-ahead bias. A high-growth
company that had a P/S of 8x in 2022 might have a P/S of 25x today —
using the 2025 value for a 2022 decision is using information that was
not available at the time.

**Mitigation:** Fundamental compatibility carries only 10% weight in the
composite score, limiting the distortion. Fundamental scores are held
constant over the entire backtest rather than updated mid-run, which at
least makes the bias consistent and predictable.

**Production fix:** Use as-reported fundamental data with a reporting lag
(Q1 earnings are not public until ~6 weeks after quarter end). Sources
like SimFin or WRDS provide this. Out of scope for this project.

---

### 6.3 Short Borrow Costs

**What it is:** Every short position in the strategy incurs a borrow rate —
the cost of borrowing shares to sell them. The transaction cost model in
`src/backtest/costs.py` uses a flat 2% annualized rate for all short legs.

**Why it matters:** Actual borrow rates vary widely by stock and market
conditions. Heavily shorted tech names can have borrow rates of 5–15%+
annualized. Using a flat 2% understates costs for these names.

**Mitigation:** The 2% assumption is conservative relative to the most
liquid names in our universe but optimistic for any stock with elevated
short interest. Net effect on Sharpe is likely small for a universe of
large-cap tech stocks. Report the assumption explicitly.

---

### 6.4 Earnings Blackout Approximation

**What it is:** The earnings blackout in `src/regime/earnings.py` uses an
end-of-quarter approximation (last 5 trading days of March, June,
September, December) rather than per-company earnings dates.

**Why it matters:** Large-cap tech companies report earnings on different
days within the quarter. Some report in January, others in February. The
approximation may block trading on days when no relevant earnings are
occurring, and may miss earnings that fall outside the approximation window.

**Production fix:** Use an earnings calendar API (e.g. Nasdaq API,
Intrinio) to get per-company earnings dates. Out of scope for this project.

---

### 6.5 Adjustment Methodology Inconsistency

**What it is:** yfinance applies split and dividend adjustments using its
own default methodology, which may differ from other data vendors.

**Why it matters:** Different adjustment methodologies can produce slightly
different return series, which affects correlation calculations and
cointegration test results.

**Mitigation:** All tickers use the same source (yfinance) with the same
adjustment methodology applied consistently, so relative comparisons
between pairs are unaffected. Absolute return levels may differ from
vendor-adjusted data.

---

## 7. Data Quality Checks

`src/data/clean.py` runs the following checks on every fetch and writes
a report to `outputs/data_quality_report.txt`.


| Check             | Rule                                        | Action on Failure             |
| ----------------- | ------------------------------------------- | ----------------------------- |
| Missing days      | Forward-fill if ≤ 1 consecutive missing day | Fill with prior day's value   |
| Extended gaps     | Flag if ≥ 2 consecutive missing days        | Log warning, do not fill      |
| Missing day count | Count missing days in trailing 90 days      | Drop ticker if > 5 missing    |
| Negative prices   | Adj close should never be negative          | Log error, drop affected rows |
| Zero volume       | Flag days with zero volume                  | Log warning, keep rows        |
| Return outliers   | Flag daily log returns beyond ±50%          | Log warning for review        |


**Always review `outputs/data_quality_report.txt` after a fresh fetch.**
Tickers that were dropped will not appear in any downstream processing.
If a key ticker was dropped unexpectedly, investigate before running the
backtest.

---

## 8. Troubleshooting

### Fetch fails with 403 or proxy error

Yahoo Finance actively rate-limits and blocks requests that look like
automated scraping. This is the most common fetch failure.

```bash
# Check if a proxy is configured in your shell
echo $HTTP_PROXY
echo $HTTPS_PROXY

# If a proxy URL is printed, unset it and retry
unset HTTP_PROXY && unset HTTPS_PROXY
uv run python scripts/01_fetch_data.py --disable-proxy
```

If the error persists, switch to a personal hotspot rather than campus WiFi.
Institutional networks commonly block financial data APIs at the firewall.

Once you have a successful fetch, keep `data/raw/` locally (it is gitignored)
and re-run fetch on any other machine that needs the same cache.

---

### Fundamentals hit Yahoo rate limits

Price downloads are batched, but fundamentals still require one Yahoo
quote-summary request per ticker. Yahoo may return:

```text
Too Many Requests. Rate limited. Try after a while.
```

When this happens, the script writes:

- `data/raw/fetch_status.json` — current stage, last ticker attempted,
next ticker to resume, completed count, and pending count
- `data/raw/fundamentals_checkpoint.parquet` — partial successful
fundamentals records

Recommended recovery:

```bash
# 1. Wait 15-30 minutes, or longer if rate limits persist.

# 2. Resume only fundamentals; do not refetch prices.
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals --resume

# 3. If limits continue, slow the request cadence.
uv run python scripts/01_fetch_data.py --disable-proxy --stage fundamentals \
  --resume --fundamentals-delay 10 --rate-limit-cooldown 1800
```

Do not repeatedly restart the full fetch. That reuses Yahoo endpoints
unnecessarily and makes rate limits more likely.

Alternatives if Yahoo fundamentals are unreliable:

- **FMP / Financial Modeling Prep:** Easy REST API for ratios and income
statement fields. Good student-project option, but check free-tier limits.
- **Alpha Vantage:** Free tier available; slower and more rate-limited, but
documented and stable for basic fundamentals.
- **Tiingo:** Cleaner API and good price data. Fundamentals availability
depends on plan.
- **Polygon.io:** Production-grade market data; fundamentals generally
require a paid plan.
- **SimFin:** Better fit for point-in-time fundamentals and avoiding
look-ahead bias, but requires schema mapping.
- **WRDS / Compustat:** Best academic source if your university provides
access. Strongest option for a rigorous backtest.
- **Scope cut:** If no reliable fundamentals source is available, set the
fundamental component weight to zero and redistribute that 10% across the
other composite score components. Document this in `docs/decisions.md`.

---

### Generating synthetic regime data for development

If Yahoo rate-limits the regime fetch and you need to unblock development
immediately, run this script to generate synthetic VIX and SPY data:

```python
import numpy as np
import pandas as pd
from pathlib import Path

start = "2021-01-01"
end   = "2024-07-01"
dates = pd.bdate_range(start, end)

rng = np.random.default_rng(42)
vix = np.clip(15 + np.cumsum(rng.normal(0, 0.3, len(dates))), 12, 45)
spy = 400 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates))))

regime = pd.DataFrame({"date": pd.to_datetime(dates), "vix": vix, "spy": spy})
Path("data/raw").mkdir(parents=True, exist_ok=True)
regime.to_parquet("data/raw/regime.parquet", index=False)
print(f"Synthetic regime: {len(regime)} rows ({dates[0].date()} → {dates[-1].date()})")
```

Save it anywhere outside `src/` and run with `uv run python <script>.py`.

**Warning:** Synthetic regime data must be replaced before the final backtest.
Synthetic data has exactly the number of business days in the date range.
Real data may have slightly fewer due to market holidays — if the row counts
match exactly, you are still on synthetic data.

---

### Backtest produces different results after re-fetching

yfinance retroactively adjusts historical prices when splits or dividends
occur. If you re-fetch after a corporate action, `adj_close` values for
historical dates will change. Delete `data/processed/` and re-run
`scripts/02_build_universe.py` after any re-fetch to ensure processed
data is consistent with the new raw data.

```bash
rm -rf data/processed/
uv run python scripts/02_build_universe.py
```

---

### A ticker is missing from processed data

Check `outputs/data_quality_report.txt` first — the ticker was likely
dropped during cleaning for having too many missing days. If the ticker
is important, either:

1. Lower `CONFIG.max_missing_days` (not recommended — may introduce
  bad data into the pipeline), or
2. Accept the drop and remove the ticker from `CANDIDATE_TICKERS`
  in `src/data/fetch.py` to keep the fetch clean

---

### `data/processed/` looks stale after changing config.py

If you change any filter threshold in `src/config.py` (price floor, ADV
minimum, SPY correlation cutoff), the processed data must be regenerated:

```bash
rm -rf data/processed/
uv run python scripts/02_build_universe.py
```

The raw data in `data/raw/` does not need to be re-fetched.

---

### Full reset

To start completely from scratch:

```bash
rm -rf data/raw/ data/processed/ outputs/
uv run python scripts/01_fetch_data.py
uv run python scripts/02_build_universe.py
```

---

## 9. Sector Map

The file `data/sector_map.py` is a hand-authored dictionary mapping each
candidate ticker to a sector label. It is committed and validated at
import time against `CANDIDATE_TICKERS`. Live `fundamentals.py` scoring
uses these labels (same sector 1.0 / cross 0.4 / unknown 0.5).

Valid labels (must match `_validate_sector_map` in that file):


| Sector label | Example tickers |
|---|---|
| Semiconductors | NVDA, AMD, INTC, AVGO |
| Enterprise Software | MSFT, ORCL, CRM, ADBE |
| Cybersecurity | FTNT, PANW, CRWD, CHKP |
| Energy | XOM, CVX, SLB, VLO |
| Financials | JPM, BAC, GS, SCHW |
| Healthcare | JNJ, UNH, ABBV, LLY |
| Consumer Staples | PG, KO, COST, WMT |
| Industrials | CAT, HON, ETN, RTX |
| Utilities | NEE, DUK, SO, AEP |
| Materials | LIN, APD, FCX, NEM |
| Consumer Discretionary | AMZN, TSLA, HD, MCD |
| Communication Services | GOOGL, META, NFLX, DIS |


**Updating the sector map:**
If a ticker is added to `CANDIDATE_TICKERS` in `src/data/fetch.py`, it
must also be added to `data/sector_map.py` before running
`scripts/02_build_universe.py`. Import of the map raises `ValueError` if
a ticker is missing or the label is not in the valid set.

---

## 10. Data Refresh Procedure

For this project, data is fetched once and not refreshed during the
development and backtesting period. If a refresh becomes necessary
(e.g. extending the backtest date range), follow this procedure:

1. Back up existing raw data if you want to preserve it:
  ```bash
   cp -r data/raw/ data/raw_backup/
  ```
2. Re-run the fetch:
  ```bash
   uv run python scripts/01_fetch_data.py
  ```
3. Review the data quality report:
  ```bash
   cat outputs/data_quality_report.txt
  ```
4. Rebuild processed data:
  ```bash
   rm -rf data/processed/
   uv run python scripts/02_build_universe.py
  ```
5. If `data/raw/` changed, re-run fetch on any other checkout that uses this cache.
6. Delete local `data/processed/` and re-run step 4 so derived files match.

