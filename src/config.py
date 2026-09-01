# Single source of truth for every tunable parameter: window lengths, percentile thresholds, composite score weights, tier pool sizes, cost assumptions, drawdown limits. Frozen dataclass — immutable at runtime. Pass CONFIG into every function that needs parameters.

"""
src/config.py — Central Strategy Configuration

Single source of truth for every tunable parameter in the ARQ pairs trading
strategy. Implemented as a frozen dataclass so parameters are immutable at
runtime — nothing can accidentally overwrite a value mid-backtest.

Usage:
    from src.config import CONFIG

    window = CONFIG.signal_window
    threshold = CONFIG.johansen_threshold

Never hardcode parameter values in module files. Always import CONFIG.
To run a sensitivity test with different parameters, instantiate a new
StrategyConfig with overrides and pass it explicitly into the function.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # -------------------------------------------------------------------------
    # Universe Selection
    # -------------------------------------------------------------------------

    # Target number of stocks in the investable universe
    universe_size: int = 100

    # Floor — if fewer than this pass filters, proceed anyway with what we have
    universe_floor: int = 60

    # Hard pre-filter thresholds
    min_price: float = 10.0                # Minimum stock price in USD
    min_adv: int = 1_000_000              # Minimum average daily volume (shares)
    min_dollar_volume: float = 25_000_000  # Minimum 30-day avg daily dollar volume ($25M)
    max_spy_correlation: float = 0.90     # Maximum correlation to SPY over 60d

    # Minimum number of distinct tech subsectors required in the universe
    min_subsectors: int = 8

    # How many days of data to check for missing values before filtering
    data_quality_window: int = 90

    # Maximum missing days allowed within the data quality window
    max_missing_days: int = 5

    # Universe reconstitution frequency in calendar days (~monthly)
    universe_refresh_days: int = 21

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    # Rolling window for the correlation matrix used in clustering
    # Intentionally decoupled from signal_window to avoid in-sample leakage
    clustering_window: int = 120

    # Range of k values evaluated by silhouette scoring
    k_min: int = 4
    k_max: int = 6

    # Buffer extra days used to survive holidays/gaps
    extbuffer_percent: float = 0.4

    # Number of random restarts per k value in K-means
    kmeans_restarts: int = 10

    # Fixed random seed for K-means reproducibility across runs
    random_seed: int = 42

    # Adjusted Rand Index threshold — flag for manual review if below this
    ari_stability_threshold: float = 0.50

    # Clustering cadence in calendar days (~quarterly)
    clustering_refresh_days: int = 63

    # -------------------------------------------------------------------------
    # Composite Scoring
    # -------------------------------------------------------------------------

    # Component weights — must sum to 1.0
    weight_correlation_stability: float = 0.20
    weight_cointegration: float = 0.25
    weight_halflife: float = 0.3
    weight_volatility: float = 0.15
    weight_fundamentals: float = 0.10

    # Cluster scoring: pairs below this absolute score are dropped.
    # Scores are weighted sums of the [0, 1] mapped component scores.
    min_composite_score: float = 0.55

    # Maximum fraction of the portfolio NAV that can be allocated to a single pair leg.
    # Caps concentration when few pairs are active (1.0 previously allowed one pair
    # to take ~90% of NAV — see diagnostics I4, WEC/XEL $85k on $100k NAV).
    # Raised 0.25 → 0.35 in Round 5: with the loss tail cut (entry band,
    # plateau stop, pre-earnings exit) per-trade expectancy is positive and
    # scales linearly — NAV $102.4k → $103.3k, Sharpe 0.33 → 0.34, max dd
    # 4.2% → 5.1%, win rate unchanged. Do not raise further without retesting
    # concentration (Round 3 I4).
    max_weight_per_pair: float = 0.35

    # Sizing divisor cap: investable capital is split across
    # min(active pairs, this) expected concurrent positions.
    # Tested at 4 in Round 3 (the book held 1–2 open pairs on 73% of in-market
    # days): concentrating capital scaled the loss tail faster than the wins
    # (full NAV $94.7k vs $97.1k) and pushed drawdowns past the halt. 10
    # reproduces the equal-split-across-active-pairs sizing, which won the
    # test matrix. max_weight_per_pair remains the hard per-pair cap.
    target_concurrent_pairs: int = 10

    # --- Correlation Stability ---
    # Minimum recent 60-day correlation — pairs below this are discarded
    # before composite scoring regardless of other scores
    min_recent_correlation: float = 0.50

    # Historical baseline window for correlation stability scoring (~1 year)
    correlation_stability_historical_window: int = 252

    # --- Cointegration (Johansen) ---
    # p-value threshold used to map scores; not a hard pair-elimination gate
    johansen_threshold: float = 0.10

    # Lookback (trading days) for the Johansen test. Longer than formation_window
    # so the test has power once warmup history is loaded.
    johansen_window: int = 252

    # Soft floor on the continuous cointegration score (1 - BH-adjusted p).
    # 0.70 ≈ adjusted p below 0.30. Raised from 0.40 in Round 3: the loss tail
    # came from pairs that broke down (stop-outs), and the floor curve was
    # monotone-better from 0.5 → 0.7 (NAV $94.6k → $99.3k, win 74% → 81%,
    # stops 14 → 8) before pair supply dried up at 0.85. Selected on
    # full-period results — see docs/diagnostics.md Round 3 caveats.
    min_cointegration_score: float = 0.70

    # Formation OLS hedge ratio must be strictly positive. Negative β implies
    # both legs would be traded in the same direction (not a spread hedge).
    min_formation_beta: float = 0.0

    # --- Spread Half-Life ---
    # Acceptable half-life range in trading days
    # Must align with time_stop_days — pairs outside this range are discarded
    halflife_min: int = 5
    halflife_max: int = 20

    # --- Volatility Compatibility ---
    # Weight on short-window volatility ratio (remaining goes to long-window)
    volatility_short_weight: float = 0.60
    volatility_long_weight: float = 0.40

    # Short and long volatility windows in trading days
    volatility_short_window: int = 20
    volatility_long_window: int = 120

    # Maximum short-window volatility ratio before pair is discarded
    max_volatility_ratio: float = 2.50

    # --- Fundamental Compatibility ---
    # Sector-compatibility scores used by the current fundamentals scorer.
    # The scorer now uses sector labels from data/sector_map.py rather than
    # yfinance fundamentals.
    same_sector_score: float = 1.0
    cross_sector_score: float = 0.4
    unknown_sector_score: float = 0.5

    # Number of pairs selected per cluster. 2 doubles portfolio capacity
    # (engine caps concurrent positions at len(active_pairs)); the
    # no-shared-tickers constraint still applies at entry.
    finalists_per_cluster: int = 2

    # -------------------------------------------------------------------------
    # Signal Generation
    # -------------------------------------------------------------------------

    # Formation window: The historical lookback used to calculate the locked
    # cointegrating vector (beta, mean, std) during the out-of-sample split.
    formation_window: int = 120

    # Signal window: Used strictly for correlation and volatility components
    # during candidate scoring. Not used for OLS regression anymore.
    signal_window: int = 60

    # Floor on formation spread σ_F when computing formation z-score
    # (spread_today − μ_F) / max(σ_F, this value); avoids exploding z when σ_F≈0.
    min_formation_spread_std: float = 1e-4

    # Hedge ratio rebalance trigger — rebalance short leg when beta drifts beyond
    beta_rebalance_threshold: float = 0.15

    # Resize leg B while a position is open when the live 60-day β drifts.
    # Disabled: the absolute 0.15 deadband on a noisy rolling β caused frequent
    # resizes that realized cash buy-high/sell-low and could flip the short
    # leg's sign (diagnostics I5, UNH/ABBV). The formation hedge is held to exit.
    rebalance_beta_intra_trade: bool = False

    # -------------------------------------------------------------------------
    # Entry and Exit Signals (Z-Score thresholds)
    # -------------------------------------------------------------------------

    # Entry: absolute Z-score of the spread must exceed this to enter a trade.
    # > +entry_zscore -> Short Spread. < -entry_zscore -> Long Spread.
    # 1.25 (was 1.5) selected by the Round 3 in-sample sweep: tighter entries
    # catch small dislocations that actually revert; wider entries (1.75-2.25)
    # caught genuinely diverging spreads and halved the win rate.
    entry_zscore: float = 1.25

    # Require a fresh threshold cross to enter: yesterday's z inside the entry
    # band, today's beyond it. Prevents entering mid-divergence on the score
    # date, where any pair already past the threshold fired immediately
    # (diagnostics RC4 — SCHW/BAC stopped out in 8 days, twice).
    entry_requires_cross: bool = True

    # Upper bound of the entry band: do not enter when |z| already exceeds
    # this. A cross that lands far beyond the entry threshold is a single-day
    # repricing (news), not a slow dislocation — Round 4: EXC/SO crossed from
    # inside the band to z=-4.9 overnight (beyond the stop itself) and lost
    # -$923 the next day; GS/MS gapped to 2.2 and stopped in 3 days. The
    # remaining distance to the stop is also too small for the risk/reward.
    # None disables the cap.
    entry_zscore_max: float | None = 2.0

    # Plateau stop: exit when the adverse z-score has been at or beyond this
    # level for stop_plateau_days consecutive trading days. Round 4: every
    # slow-bleed stop-out (TMO/DHR, NOW/ANET, WEC/XEL, SO/WEC) lingered at
    # z 2.4-3.3 for 1-4 weeks without re-approaching the mean before finally
    # breaching 3.5 — the reversion premise was already dead. Exiting after a
    # sustained plateau realizes a ~1σ smaller loss than the hard stop.
    # stop_plateau_days = 0 disables.
    stop_plateau_zscore: float = 2.75
    stop_plateau_days: int = 3

    # Take profit: close position when absolute Z-score reverts back towards
    # the mean. 1.0 (was 0.5) selected by the Round 3 in-sample sweep: taking
    # the first ~0.25σ of reversion wins far more often than holding for a
    # full retrace (IS win rate 80% vs 46% at the old 1.5/0.5 shape).
    take_profit_zscore: float = 1.0

    # Stop loss: close immediately if absolute Z-score stretches beyond this limit
    stop_loss_zscore: float = 3.5

    # Momentum filter parameters (prevents stepping in front of freight trains)
    momentum_window: int = 14
    momentum_threshold: float = 0.15

    # Time stop: maximum days to hold a position before force closing.
    time_stop_days: int = 50

    # Per-pair dollar loss cap: force-close a position when its unrealized
    # P&L falls below -(this fraction) × dollar allocation. Disabled: the
    # Round 3 test matrix (caps 2%/3%/5% vs none) showed every cap LOWERED
    # both win rate and NAV — dips of 2-5% usually revert, so the cap
    # converts temporary drawdowns into realized losses. Kept as an optional
    # risk control for live trading.
    max_pair_loss_pct: float | None = None

    # Cooldown after a STOP_LOSS before the same pair may re-enter.
    # Uses trading-day steps in the engine loop.
    pair_stop_cooldown_days: int = 20

    # -------------------------------------------------------------------------
    # Regime Filters
    # -------------------------------------------------------------------------

    # --- VIX Filter (portfolio level) ---

    # VIX level that blocks all new position entries across the portfolio
    vix_entry_block: float = 28.0

    # VIX level required to resume entries (must hold for vix_resume_days)
    vix_resume: float = 25.0

    # Number of consecutive trading days VIX must stay below vix_resume
    vix_resume_days: int = 5

    # --- Earnings Blackout (pair level) ---

    # Trading days before end of quarter to begin blackout
    earnings_blackout_days_before: int = 5

    # Trading days after end of quarter before entries resume
    earnings_blackout_days_after: int = 1

    # --- Defensive pre-earnings exit (real earnings dates) ---
    # Force-close an open pair this many trading days before either leg
    # reports earnings, but ONLY when the position is already losing
    # (adverse formation z >= earnings_exit_min_adverse_z). Rationale
    # (Round 5 data): the two largest losses in the book were single-day
    # earnings gaps through the stop (SCHW -10% on 2024-07-16 earnings,
    # -$4.0k; VLO/CVX Oct-Nov 2024 reports, -$1.6k) — both positions were
    # already >1.75σ adverse going into the print. Winning or near-flat
    # positions are held through earnings: entries near earnings resolved
    # in our favor repeatedly (DE/HON +$963, ADSK/WDAY +$775, TXN/QCOM
    # +$621), so an unconditional exit or entry blackout gives the edge
    # back. Requires data/raw/earnings.parquet (scripts/01 --stage
    # earnings); silently disabled when the file is missing. 0 disables.
    earnings_exit_days_before: int = 2
    earnings_exit_min_adverse_z: float = 1.75

    # -------------------------------------------------------------------------
    # Position Sizing
    # -------------------------------------------------------------------------

    # Starting capital for the backtest simulation in USD
    initial_capital: float = 100_000.0

    # Target cash buffer to hold uninvested
    cash_buffer_pct: float = 0.10

    # Maximum gross leverage (long exposure + short exposure) / NAV
    max_gross_leverage: float = 2.0

    # Maximum fraction of active pairs from any single subsector
    max_subsector_concentration: float = 0.35

    # Maximum concurrent pairs from the same cluster
    max_same_cluster_pairs: int = 2

    # -------------------------------------------------------------------------
    # Drawdown Controls
    # -------------------------------------------------------------------------

    # Weekly drawdown level that triggers position size reduction on new entries
    drawdown_reduce_threshold: float = 0.05   # 5% from prior week close

    # Fraction to reduce new entry sizes when soft threshold is hit
    drawdown_reduce_factor: float = 0.50      # New entries at 50% normal size

    # Rolling peak drawdown that halts all new entries and trims existing positions
    drawdown_halt_threshold: float = 0.10     # 10% from rolling peak

    # Fraction to which existing positions are reduced when halt is triggered
    drawdown_trim_factor: float = 0.25        # Trim to 25% of current size

    # Days over which to execute the trim (to avoid market impact)
    drawdown_trim_days: int = 5

    # Legacy recovery condition (within this pct of the triggering peak).
    # No longer used for halt release: the old rule was unreachable — with all
    # positions trimmed and entries blocked, NAV could never climb back toward
    # the all-time peak, deadlocking the strategy (no entries after 2023-03-09).
    # Kept for reference / potential reuse in reporting.
    drawdown_recovery_threshold: float = 0.05

    # Consecutive non-losing days required after the trim completes before the
    # halt releases. On release the rolling peak resets to current NAV, so the
    # halt acts as a ~2-week circuit breaker rather than a permanent stop.
    drawdown_recovery_days: int = 5

    # -------------------------------------------------------------------------
    # Transaction Costs
    # -------------------------------------------------------------------------

    # Commission per share in USD (both legs)
    commission_per_share: float = 0.005

    # Slippage in basis points for liquid stocks (ADV > 5M shares)
    slippage_bps_liquid: float = 5.0

    # Slippage in basis points for medium liquidity stocks (1M-5M ADV)
    slippage_bps_medium: float = 10.0

    # ADV threshold separating liquid from medium liquidity in shares
    liquidity_threshold_adv: int = 5_000_000

    # Bid-ask spread in basis points for stocks priced above $30
    bid_ask_bps_high_price: float = 2.0

    # Bid-ask spread in basis points for stocks priced between $10 and $30
    bid_ask_bps_low_price: float = 5.0

    # Price threshold separating bid-ask spread tiers in USD
    bid_ask_price_threshold: float = 30.0

    # Flat annualized short borrow cost applied to all short legs
    # Replace with per-stock broker rates if available before live deployment
    short_borrow_annual: float = 0.02        # 2% per year

    # Minimum expected profit as a multiple of round-trip cost to enter a trade
    min_profit_to_cost_ratio: float = 2.0

    # -------------------------------------------------------------------------
    # Backtesting
    # -------------------------------------------------------------------------

    # Hard start date for the backtest. If None, uses the earliest available data.
    # Set this to "2022-11-01" to easily restrict a 7-year dataset down to 3.5 years.
    backtest_start_date: str | None = "2022-01-01"

    # Hard end date for the backtest. If None, uses all available data.
    # Set this to "2023-12-31" to easily restrict the dataset to end in 2023.
    backtest_end_date: str | None = "2024-12-31"


    # Fraction of the full dataset held out as out-of-sample
    # Portion of dataset to hold out for walk-forward validation (e.g. 0.20 = 20%)
    oos_fraction: float = 0.30
    
    # If True, the engine will only loop over the OOS timeline (bypassing the IS timeline)
    run_oos_only: bool = False

    # Partial fill threshold — cancel long leg if short leg fills below this
    partial_fill_threshold: float = 0.95     # 95% of intended short quantity


# -----------------------------------------------------------------------------
# Singleton instance — import this throughout the codebase
# -----------------------------------------------------------------------------

CONFIG = StrategyConfig()


# -----------------------------------------------------------------------------
# Validation — catches misconfiguration at import time
# -----------------------------------------------------------------------------

def _validate_config(cfg: StrategyConfig) -> None:
    """Validate internal consistency of config on startup."""

    weight_sum = (
        cfg.weight_correlation_stability
        + cfg.weight_cointegration
        + cfg.weight_halflife
        + cfg.weight_volatility
        + cfg.weight_fundamentals
    )
    assert abs(weight_sum - 1.0) < 1e-9, (
        f"Composite score weights must sum to 1.0, got {weight_sum:.4f}"
    )


    assert cfg.time_stop_days >= cfg.halflife_max * 1.5, (
        f"time_stop_days ({cfg.time_stop_days}) must be at least 1.5x halflife_max "
        f"({cfg.halflife_max}) to allow pairs enough time to revert"
    )

    assert cfg.take_profit_zscore < cfg.entry_zscore < cfg.stop_loss_zscore, (
        "Z-score thresholds must logically cascade: take_profit < entry < stop_loss"
    )

    if cfg.entry_zscore_max is not None:
        assert cfg.entry_zscore < cfg.entry_zscore_max <= cfg.stop_loss_zscore, (
            f"entry_zscore_max ({cfg.entry_zscore_max}) must lie between "
            f"entry_zscore ({cfg.entry_zscore}) and stop_loss_zscore "
            f"({cfg.stop_loss_zscore})"
        )

    if cfg.stop_plateau_days > 0:
        assert cfg.entry_zscore < cfg.stop_plateau_zscore < cfg.stop_loss_zscore, (
            f"stop_plateau_zscore ({cfg.stop_plateau_zscore}) must lie between "
            f"entry_zscore ({cfg.entry_zscore}) and stop_loss_zscore "
            f"({cfg.stop_loss_zscore})"
        )

    assert cfg.vix_resume < cfg.vix_entry_block, (
        f"vix_resume ({cfg.vix_resume}) must be less than "
        f"vix_entry_block ({cfg.vix_entry_block})"
    )

    assert cfg.volatility_short_weight + cfg.volatility_long_weight == 1.0, (
        "Volatility window weights must sum to 1.0"
    )

    assert cfg.min_formation_spread_std > 0, (
        "min_formation_spread_std must be positive (used as σ floor in formation z)"
    )


_validate_config(CONFIG)
