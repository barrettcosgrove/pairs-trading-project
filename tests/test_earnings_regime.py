"""
tests/test_earnings_regime.py - Unit tests for src/regime/earnings.py
earnings_within (real earnings-date proximity used by the defensive
pre-earnings exit).

Synthetic calendars only — no network access, no parquet reads.
"""

from datetime import date

import pandas as pd

from src.regime.earnings import earnings_within

TRADING_DAYS = pd.bdate_range("2024-07-01", "2024-07-31")
EARNINGS = {"SCHW": [pd.Timestamp("2024-07-16")]}


def test_earnings_inside_horizon():
    # 2024-07-12 is two trading days before the 07-16 report (15th, 16th).
    assert earnings_within("SCHW", date(2024, 7, 12), 2, TRADING_DAYS, EARNINGS)


def test_earnings_beyond_horizon():
    # 2024-07-05 is seven trading days before the report — outside n=2.
    assert not earnings_within("SCHW", date(2024, 7, 5), 2, TRADING_DAYS, EARNINGS)


def test_report_day_itself_not_flagged():
    # The gap has already happened by the close of the report day.
    assert not earnings_within("SCHW", date(2024, 7, 16), 2, TRADING_DAYS, EARNINGS)


def test_unknown_ticker_returns_false():
    assert not earnings_within("ZZZZ", date(2024, 7, 12), 2, TRADING_DAYS, EARNINGS)


def test_weekend_report_attributed_forward():
    # Report dated Saturday 07-20; evaluated Friday 07-19 with n=1 the next
    # trading day is Monday 07-22, which the Saturday date falls before.
    earnings = {"AAA": [pd.Timestamp("2024-07-20")]}
    assert earnings_within("AAA", date(2024, 7, 19), 1, TRADING_DAYS, earnings)
