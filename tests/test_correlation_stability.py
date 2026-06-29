from datetime import date

import pandas as pd
import pytest

from src.scoring.correlation_stability import score, score_candidate_pairs


@pytest.fixture
def stable_pair_returns():
    dates = pd.bdate_range("2023-01-02", periods=300)

    series_a = [0.01 if i % 2 == 0 else -0.01 for i in range(300)]
    series_b = [0.011 if i % 2 == 0 else -0.009 for i in range(300)]

    rows = []
    for dt, a_ret, b_ret in zip(dates, series_a, series_b):
        rows.append({"date": dt, "ticker": "AAA", "log_return": a_ret})
        rows.append({"date": dt, "ticker": "BBB", "log_return": b_ret})

    return pd.DataFrame(rows)


@pytest.fixture
def weak_recent_pair_returns():
    dates = pd.bdate_range("2023-01-02", periods=300)

    series_a = [0.01 if i % 2 == 0 else -0.01 for i in range(300)]
    series_b = [0.011 if i % 2 == 0 else -0.009 for i in range(240)]

    # Last 60 days: break the relationship
    tail_b = [0.01 if i % 3 == 0 else 0.0 for i in range(60)]
    series_b.extend(tail_b)

    rows = []
    for dt, a_ret, b_ret in zip(dates, series_a, series_b):
        rows.append({"date": dt, "ticker": "AAA", "log_return": a_ret})
        rows.append({"date": dt, "ticker": "BBB", "log_return": b_ret})

    return pd.DataFrame(rows)


@pytest.fixture
def insufficient_history_returns():
    dates = pd.bdate_range("2024-01-02", periods=100)

    rows = []
    for i, dt in enumerate(dates):
        rows.append({"date": dt, "ticker": "AAA", "log_return": 0.01 if i % 2 == 0 else -0.01})
        rows.append({"date": dt, "ticker": "BBB", "log_return": 0.011 if i % 2 == 0 else -0.009})

    return pd.DataFrame(rows)


def test_score_returns_value_in_bounds(stable_pair_returns):
    result = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=stable_pair_returns,
        as_of=date(2024, 2, 23),
    )

    assert 0.0 <= result <= 1.0


def test_score_high_for_stable_pair(stable_pair_returns):
    result = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=stable_pair_returns,
        as_of=date(2024, 2, 23),
    )

    assert result > 0.8


def test_score_zero_when_recent_correlation_too_low(weak_recent_pair_returns):
    result = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=weak_recent_pair_returns,
        as_of=date(2024, 2, 23),
    )

    assert result == 0.0


def test_score_zero_when_not_enough_history(insufficient_history_returns):
    result = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=insufficient_history_returns,
        as_of=date(2024, 5, 31),
    )

    assert result == 0.0


def test_score_respects_as_of_and_ignores_future_data(stable_pair_returns):
    result_early = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=stable_pair_returns,
        as_of=date(2023, 10, 2),
    )
    result_late = score(
        ticker_a="AAA",
        ticker_b="BBB",
        returns=stable_pair_returns,
        as_of=date(2024, 2, 23),
    )

    assert 0.0 <= result_early <= 1.0
    assert 0.0 <= result_late <= 1.0


def test_score_candidate_pairs_adds_score_column(stable_pair_returns):
    candidate_pairs = pd.DataFrame(
        [
            {"ticker_a": "AAA", "ticker_b": "BBB", "cluster_id": 0},
        ]
    )

    result = score_candidate_pairs(
        candidate_pairs=candidate_pairs,
        returns=stable_pair_returns,
        as_of=date(2024, 2, 23),
    )

    assert "correlation_stability_score" in result.columns
    assert len(result) == 1
    assert result.loc[0, "ticker_a"] == "AAA"
    assert result.loc[0, "ticker_b"] == "BBB"
    assert result.loc[0, "cluster_id"] == 0
    assert 0.0 <= result.loc[0, "correlation_stability_score"] <= 1.0


def test_score_candidate_pairs_validates_required_columns(stable_pair_returns):
    bad_pairs = pd.DataFrame(
        [
            {"ticker_a": "AAA", "ticker_b": "BBB"},
        ]
    )

    with pytest.raises(ValueError, match="required columns"):
        score_candidate_pairs(
            candidate_pairs=bad_pairs,
            returns=stable_pair_returns,
            as_of=date(2024, 2, 23),
        )
