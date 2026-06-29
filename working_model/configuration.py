"""Centralized configuration for the working_model prototype."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkingModelConfig:
    """Configuration for data fetch, walk-forward backtest, and portfolio accounting."""

    # Fetch/cache request bounds (used for parquet key and yfinance download range).
    fetch_start_date: str = "2018-01-01"
    fetch_end_date: str = "2025-09-01"

    # Optional in-memory panel slice after loading parquet/yfinance.
    panel_use_start: str | None = None
    panel_use_end: str | None = None

    # Walk-forward evaluation window.
    backtest_start: str = "2023-01-01"
    backtest_end: str = "2025-06-04"

    # Discovery cadence.
    formation_days: int = 252
    rescore_freq_trading_days: int = 21
    min_coint_history: int = 200

    # Clustering: fixed KMeans k=n_clusters, or optional silhouette k in [cluster_k_min, cluster_k_max].
    use_silhouette_k_selection: bool = False  # If False, always use ``n_clusters`` (default 5).
    cluster_k_min: int = 4
    cluster_k_max: int = 6
    n_clusters: int = 5  # Used when silhouette is off; also fallback if silhouette scan fails.
    kmeans_n_init: int = 10
    kmeans_random_seed: int = 42

    # Universe hard filters (applied point-in-time at each walk-forward rescore).
    min_price: float = 10.0
    min_adv: int = 1_000_000
    min_dollar_volume: float = 25_000_000
    liquidity_window: int = 30
    max_spy_correlation: float = 0.90
    spy_correlation_window: int = 60
    spy_min_observations: int = 30
    spy_ticker: str = "SPY"

    # Signal/risk controls.
    half_life_min_days: int = 5
    half_life_max_days: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_loss_z: float = 3.0
    max_holding_multiplier: float = 3.0

    # Regime: VIX blocks new entries only (exits and risk exits always allowed).
    # Decision uses prior trading day's VIX close vs equity calendar (shift(1); no same-day peek).
    use_vix_filter: bool = True
    vix_ticker: str = "^VIX"
    vix_entry_block: float = 28.0
    vix_resume: float = 25.0
    vix_resume_days: int = 5

    # Portfolio accounting controls.
    initial_capital: float = 100_000.0
    max_active_pairs: int = 5
    target_gross_per_pair_pct: float = 0.20
    max_gross_exposure_pct: float = 1.00
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    short_borrow_annual: float = 0.02
    trading_days_per_year: int = 252

    # Universe for this prototype.
    tickers: list[str] = field(
        default_factory=lambda: [
            "NVDA",
            "AMD",
            "INTC",
            "QCOM",
            "AVGO",
            "TXN",
            "MU",
            "AMAT",
            "MSFT",
            "CRM",
            "NOW",
            "ADSK",
            "CDNS",
            "SNPS",
            "WDAY",
            "ORCL",
            "ADBE",
            "ANET",
            "FTNT",
            "PANW",
            "CRWD",
            "CHKP",
            "XOM",
            "CVX",
            "COP",
            "EOG",
            "SLB",
            "MPC",
            "PSX",
            "VLO",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "C",
            "BLK",
            "AXP",
            "SCHW",
            "JNJ",
            "UNH",
            "PFE",
            "ABBV",
            "MRK",
            "TMO",
            "ABT",
            "DHR",
            "LLY",
            "BSX",
            "PG",
            "KO",
            "PEP",
            "COST",
            "WMT",
            "PM",
            "CL",
            "GIS",
            "CAT",
            "DE",
            "HON",
            "GE",
            "LMT",
            "UPS",
            "FDX",
            "EMR",
            "ETN",
            "RTX",
            "NEE",
            "DUK",
            "SO",
            "AEP",
            "EXC",
            "D",
            "XEL",
            "WEC",
            "LIN",
            "APD",
            "SHW",
            "ECL",
            "NEM",
            "FCX",
            "AMZN",
            "TSLA",
            "HD",
            "MCD",
            "NKE",
            "SBUX",
            "LOW",
            "TJX",
            "GOOGL",
            "META",
            "NFLX",
            "DIS",
            "CMCSA",
            "T",
        ]
    )


CONFIG = WorkingModelConfig()
