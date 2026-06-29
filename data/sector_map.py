# Hardcoded dictionary mapping each ticker to its GICS subsector. Manually maintained. Used by the universe filter and the subsector concentration constraint in risk management.
"""
data/sector_map.py — Ticker → Subsector Map

Hand-authored mapping from each candidate ticker to its primary tech subsector.
Consumed by src/universe/filter.py (sector gate) and src/backtest/portfolio.py
(subsector concentration limit).

Valid subsector values:
    semiconductors, cloud_infrastructure, cybersecurity, enterprise_software,
    consumer_hardware, gaming, social_media, ai_infrastructure, fintech,
    ecommerce_tech

If a new ticker is added to CANDIDATE_TICKERS in src/data/fetch.py it must
also be added here, or scripts/02_build_universe.py will raise a ValueError
at import time.
"""

# Sector mapping based on your CANDIDATE_TICKERS
SECTOR_MAP: dict[str, str] = {
    # Semiconductors
    "NVDA": "Semiconductors", "AMD": "Semiconductors", "INTC": "Semiconductors",
    "QCOM": "Semiconductors", "AVGO": "Semiconductors", "TXN": "Semiconductors",
    "MU": "Semiconductors", "AMAT": "Semiconductors",
    
    # Cloud / Enterprise Software
    "MSFT": "Enterprise Software", "CRM": "Enterprise Software", "NOW": "Enterprise Software",
    "ADSK": "Enterprise Software", "CDNS": "Enterprise Software", "SNPS": "Enterprise Software",
    "WDAY": "Enterprise Software", "ORCL": "Enterprise Software", "ADBE": "Enterprise Software",
    "ANET": "Enterprise Software",
    
    # Cybersecurity
    "FTNT": "Cybersecurity", "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "CHKP": "Cybersecurity",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "BLK": "Financials", "AXP": "Financials",
    "SCHW": "Financials",
    
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "LLY": "Healthcare", "BSX": "Healthcare",
    
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "CL": "Consumer Staples", "GIS": "Consumer Staples",
    
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "LMT": "Industrials", "UPS": "Industrials", "FDX": "Industrials", "EMR": "Industrials",
    "ETN": "Industrials", "RTX": "Industrials",
    
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    "EXC": "Utilities", "D": "Utilities", "XEL": "Utilities", "WEC": "Utilities",
    
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "NEM": "Materials", "FCX": "Materials",
    
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    
    # Communication Services
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services", "T": "Communication Services",
}


# ---------------------------------------------------------------------------
# Validation — runs at import time
# ---------------------------------------------------------------------------

def _validate_sector_map() -> None:
    """Raise ValueError if any CANDIDATE_TICKER is missing from SECTOR_MAP."""
    import sys
    import os

    # Locate the project root relative to this file (data/sector_map.py → root)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from src.data.fetch import CANDIDATE_TICKERS

    missing = [t for t in CANDIDATE_TICKERS if t not in SECTOR_MAP]
    if missing:
        raise ValueError(
            f"sector_map.py is missing entries for {len(missing)} ticker(s): "
            f"{', '.join(sorted(missing))}. "
            "Add each ticker to SECTOR_MAP before running scripts/02_build_universe.py."
        )

    valid_subsectors = {
        "Semiconductors", "Enterprise Software", "Cybersecurity",
        "Energy", "Financials", "Healthcare",
        "Consumer Staples", "Industrials", "Utilities", "Materials", 
        "Consumer Discretionary", "Communication Services"
    }
    invalid = {
        t: v for t, v in SECTOR_MAP.items() if v not in valid_subsectors
    }
    if invalid:
        raise ValueError(
            f"sector_map.py has {len(invalid)} ticker(s) with invalid subsector values: "
            f"{invalid}. Valid values are: {sorted(valid_subsectors)}"
        )


_validate_sector_map()
