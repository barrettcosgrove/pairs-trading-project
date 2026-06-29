"""
src/scoring/fundamentals.py - Sector Compatibility Scoring

Scores whether two candidate-pair tickers are economically compatible based on
their mapped labels in data/sector_map.py. The score is intended to replace
the original yfinance-based fundamentals component in composite.py.
"""
import logging

import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)

REQUIRED_CANDIDATE_PAIR_COLUMNS = {"ticker_a", "ticker_b", "cluster_id"}

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

def score(ticker_a: str, ticker_b: str) -> float:
    """
    Score whether two tickers belong to the same mapped sector label.

    Args:
        ticker_a: First ticker in the candidate pair.
        ticker_b: Second ticker in the candidate pair.

    Returns:
        Float score in [0, 1]. Returns CONFIG.same_sector_score when both
        tickers share the same mapped label, CONFIG.cross_sector_score when
        they differ, and CONFIG.unknown_sector_score when one or both labels
        are missing.
    """
    sector_a = _get_sector_label(ticker_a)
    sector_b = _get_sector_label(ticker_b)
    return _score_from_labels(sector_a=sector_a, sector_b=sector_b)


def score_candidate_pairs(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Score sector compatibility across a batch of candidate pairs.

    Args:
        candidate_pairs: DataFrame with columns [ticker_a, ticker_b, cluster_id].

    Returns:
        Copy of candidate_pairs with an added fundamentals_score column that
        represents sector compatibility.
    """
    _validate_candidate_pairs(candidate_pairs)

    scored_pairs = candidate_pairs.copy()
    scored_pairs["fundamentals_score"] = scored_pairs.apply(
        lambda row: score(
            ticker_a=row["ticker_a"],
            ticker_b=row["ticker_b"],
        ),
        axis=1,
    )

    logger.info(
        "Scored sector compatibility for %d candidate pairs",
        len(scored_pairs),
    )

    return scored_pairs


def _validate_candidate_pairs(candidate_pairs: pd.DataFrame) -> None:
    """
    Validate the candidate-pair batch input required for scoring.

    Args:
        candidate_pairs: Candidate pair DataFrame to validate.

    Returns:
        None. Raises ValueError if required columns are missing.
    """
    missing_columns = REQUIRED_CANDIDATE_PAIR_COLUMNS.difference(candidate_pairs.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Candidate pair DataFrame is missing required columns: {missing_list}"
        )


def _get_sector_label(ticker: str) -> str | None:
    """
    Look up the mapped sector label for a ticker.

    Args:
        ticker: Ticker symbol to look up.

    Returns:
        Sector label string if the ticker exists in SECTOR_MAP, otherwise None.
    """
    return SECTOR_MAP.get(ticker)


def _score_from_labels(sector_a: str | None, sector_b: str | None) -> float:
    """
    Convert two mapped sector labels into a compatibility score.

    Args:
        sector_a: Mapped sector label for ticker A.
        sector_b: Mapped sector label for ticker B.

    Returns:
        Float score in [0, 1] using config-driven weights for same-sector,
        cross-sector, and unknown-sector cases.
    """
    if sector_a is None or sector_b is None:
        logger.debug(
            "Missing sector label for pair (%s, %s); using unknown-sector score",
            sector_a,
            sector_b,
        )
        return float(CONFIG.unknown_sector_score)

    if sector_a == sector_b:
        return float(CONFIG.same_sector_score)

    return float(CONFIG.cross_sector_score)
