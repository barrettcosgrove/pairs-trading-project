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
