import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

print("=" * 55)
print("  ARQ PRE-INTEGRATION SANITY CHECK")
print("=" * 55)
passed = []
failed = []

def check(name, fn):
    try:
        fn()
        print(f"  [OK]  {name}")
        passed.append(name)
    except Exception as e:
        print(f"  [!!]  {name}")
        print(f"        {e}")
        failed.append(name)

# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------
def test_config():
    from src.config import CONFIG
    assert CONFIG.signal_window == 60
    assert abs(CONFIG.high_pool_pct + CONFIG.low_pool_pct + CONFIG.cash_buffer_pct - 1.0) < 1e-9
    assert CONFIG.halflife_max == CONFIG.time_stop_days

check("config.py loads and validates", test_config)

# ------------------------------------------------------------------
# 2. Sector map
# ------------------------------------------------------------------
def test_sector_map():
    from data.sector_map import SECTOR_MAP
    from src.data.fetch import CANDIDATE_TICKERS
    missing = [t for t in CANDIDATE_TICKERS if t not in SECTOR_MAP]
    assert not missing, f"Missing from sector_map: {missing}"
    assert "MGAM"  not in SECTOR_MAP, "MGAM should be removed"
    assert "NEVER" not in SECTOR_MAP, "NEVER should be removed"

check("sector_map.py all tickers mapped, MGAM/NEVER removed", test_sector_map)

# ------------------------------------------------------------------
# 3. Raw data files exist
# ------------------------------------------------------------------
def test_raw_prices():
    p = pd.read_parquet("data/raw/prices.parquet")
    assert "ticker"    in p.columns
    assert "adj_close" in p.columns
    assert "volume"    in p.columns
    assert len(p) > 50000
    assert p["ticker"].nunique() >= 80

check("data/raw/prices.parquet exists with correct schema", test_raw_prices)

def test_raw_regime():
    r = pd.read_parquet("data/raw/regime.parquet")
    assert "vix"  in r.columns
    assert "spy"  in r.columns
    assert "date" in r.columns
    assert len(r) > 500

check("data/raw/regime.parquet exists with correct schema", test_raw_regime)

# ------------------------------------------------------------------
# 4. load.py
# ------------------------------------------------------------------
def test_load_prices():
    from src.data.load import load_prices
    df = load_prices(date(2024, 1, 1), date(2024, 3, 31))
    assert list(df.columns) == ["date", "ticker", "adj_close", "volume"]
    assert df["adj_close"].isna().sum() == 0
    assert len(df) > 0

check("load_prices() correct columns, no NaN", test_load_prices)

def test_load_returns():
    from src.data.load import load_returns
    df = load_returns(date(2024, 1, 1), date(2024, 3, 31))
    assert list(df.columns) == ["date", "ticker", "log_return"]
    assert df["log_return"].isna().sum() == 0
    assert df["date"].is_monotonic_increasing

check("load_returns() correct columns, no NaN, sorted", test_load_returns)

def test_load_vix():
    from src.data.load import load_vix
    s = load_vix(date(2024, 1, 1), date(2024, 3, 31))
    assert s.name == "vix"
    assert len(s) > 0
    assert s.isna().sum() == 0

check("load_vix() correct name, no NaN", test_load_vix)

# ------------------------------------------------------------------
# 5. clean.py
# ------------------------------------------------------------------
def test_clean():
    from src.data.clean import clean_prices
    prices = pd.read_parquet("data/raw/prices.parquet")
    returns, dropped, flagged = clean_prices(prices)
    assert list(returns.columns) == ["ticker", "date", "log_return"]
    assert returns["log_return"].isna().sum() == 0
    assert returns["ticker"].nunique() >= 80
    assert isinstance(dropped, dict)
    assert isinstance(flagged, pd.DataFrame)

check("clean_prices() correct output shape and types", test_clean)

def test_returns_parquet():
    df = pd.read_parquet("data/processed/returns.parquet")
    assert list(df.columns) == ["ticker", "date", "log_return"]
    assert df["log_return"].isna().sum() == 0

check("data/processed/returns.parquet exists with correct schema", test_returns_parquet)

# ------------------------------------------------------------------
# 6. filter.py structural test
# ------------------------------------------------------------------
def test_filter_structural():
    from src.universe.filter import build_universe_history
    from src.data.load import load_prices, load_returns

    prices  = load_prices(date(2023, 1, 1), date(2024, 1, 1))
    returns = load_returns(date(2023, 1, 1), date(2024, 1, 1))

    idx = pd.date_range("2023-01-01", "2024-01-01", freq="B")
    spy_returns = pd.Series(
        np.random.normal(0.0003, 0.01, len(idx)),
        index=idx,
        name="spy",
    )

    recon_dates = [date(2023, 6, 1)]
    history = build_universe_history(prices, returns, spy_returns, recon_dates)

    assert "date"           in history.columns
    assert "ticker"         in history.columns
    assert "passed_filters" in history.columns
    assert len(history) > 0
    passing = history[history["passed_filters"]]
    assert len(passing) > 0, "Expected at least some tickers to pass filters"

check("filter.py runs, correct schema, some tickers pass", test_filter_structural)

# ------------------------------------------------------------------
# 7. Data contracts
# ------------------------------------------------------------------
def test_data_contracts():
    from src.data.load import load_prices, load_returns, load_vix

    p = load_prices(date(2024, 1, 1), date(2024, 6, 30))
    r = load_returns(date(2024, 1, 1), date(2024, 6, 30))
    v = load_vix(date(2024, 1, 1), date(2024, 6, 30))

    price_tickers  = set(p["ticker"].unique())
    return_tickers = set(r["ticker"].unique())
    assert return_tickers.issubset(price_tickers), \
        "Returns contain tickers not in prices"

    price_dates = set(p["date"].dt.date)
    vix_dates   = set(v.index.date)
    overlap     = price_dates & vix_dates
    assert len(overlap) > 50, \
        f"Insufficient date overlap between prices and VIX: {len(overlap)} days"

check("Data contracts tickers and dates align across modules", test_data_contracts)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 55)
print(f"  {len(passed)} passed    {len(failed)} failed")
print("=" * 55)
if failed:
    print()
    print("  Fix failed checks before writing 02_build_universe.py")
else:
    print()
    print("  All checks passed.")
    print("  Ready to write 02_build_universe.py")
print()