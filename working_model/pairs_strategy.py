import argparse
import itertools
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt

from price_cache import (
    default_cache_raw_dir,
    format_price_load_report,
    format_vix_load_report,
    load_or_fetch_prices,
    load_or_fetch_vix,
    load_or_fetch_volumes,
)
from configuration import CONFIG

# --- Phase 1: Data Acquisition ---
def download_data(
    tickers,
    start_date,
    end_date,
    *,
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
    use_start: str | None = None,
    use_end: str | None = None,
):
    """Load adjusted close panel from cache or yfinance (same cleaning as ``load_or_fetch_prices``).

    Args:
        tickers: Iterable of ticker symbols.
        start_date: Inclusive start (string or datetime-like).
        end_date: Exclusive end for yfinance (string or datetime-like).
        cache_raw_dir: Optional directory for parquet cache (defaults to ``working_model/cache/raw``).
        force_refresh: If True, refetch and overwrite cache.
        use_start: Optional inclusive first date of the working panel (row subset after load).
        use_end: Optional inclusive last date of the working panel.

    Returns:
        Wide DataFrame of closes indexed by date.
    """
    df, _meta = load_or_fetch_prices(
        list(tickers),
        str(start_date),
        str(end_date),
        cache_raw_dir=cache_raw_dir,
        force_refresh=force_refresh,
        use_start=use_start,
        use_end=use_end,
    )
    return df

def calculate_returns(prices):
    """Compute simple daily percentage returns and drop the initial NaN row.

    Args:
        prices: Wide price panel.

    Returns:
        DataFrame of returns aligned to prices columns.
    """
    returns = prices.pct_change().dropna()
    return returns

# --- Phase 2: K-Means Clustering ---


def silhouette_scan_kmeans(
    returns: pd.DataFrame,
    *,
    k_min: int,
    k_max: int,
    n_init: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit K-means for each k and record mean silhouette on scaled return rows.

    Args:
        returns: Wide simple returns; each column is a ticker.
        k_min: Minimum k (inclusive), at least 2.
        k_max: Maximum k (inclusive); capped by n_stocks - 1.
        n_init: KMeans n_init passes.
        random_state: RNG seed for reproducibility.

    Returns:
        DataFrame with columns ``k``, ``silhouette``, ``degenerate`` (True if k skipped).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns.T)
    n_stocks = scaled.shape[0]
    lo = max(2, int(k_min))
    hi = min(int(k_max), n_stocks - 1)
    rows: list[dict] = []
    if hi < lo:
        return pd.DataFrame(columns=["k", "silhouette", "degenerate"])
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            rows.append({"k": k, "silhouette": np.nan, "degenerate": True})
            continue
        try:
            sil = float(silhouette_score(scaled, labels, metric="euclidean"))
        except ValueError:
            rows.append({"k": k, "silhouette": np.nan, "degenerate": True})
            continue
        rows.append({"k": k, "silhouette": sil, "degenerate": False})
    return pd.DataFrame(rows)


def cluster_stocks(
    returns: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    use_silhouette_k_selection: bool | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
    kmeans_n_init: int | None = None,
    random_state: int | None = None,
    verbose: bool = True,
) -> pd.Series:
    """Cluster stocks by standardized return vectors; optionally pick k via silhouette.

    When ``use_silhouette_k_selection`` is True, k is chosen as the value in
    ``[k_min, k_max]`` (clipped to valid range for n_stocks) that maximizes
    mean silhouette on the same scaled feature space KMeans uses.

    Args:
        returns: Wide simple returns; each column is a ticker.
        n_clusters: Fixed k when silhouette selection is off or as fallback.
        use_silhouette_k_selection: If None, uses ``CONFIG.use_silhouette_k_selection``.
        k_min: Minimum k for silhouette search; default from ``CONFIG.cluster_k_min``.
        k_max: Maximum k for silhouette search; default from ``CONFIG.cluster_k_max``.
        kmeans_n_init: KMeans restarts; default from ``CONFIG.kmeans_n_init``.
        random_state: Seed; default from ``CONFIG.kmeans_random_seed``.
        verbose: If True, log window and chosen k / silhouette.

    Returns:
        Series mapping ticker to integer cluster id.
    """
    n_fallback = int(CONFIG.n_clusters if n_clusters is None else n_clusters)
    use_sil = CONFIG.use_silhouette_k_selection if use_silhouette_k_selection is None else use_silhouette_k_selection
    lo = int(CONFIG.cluster_k_min if k_min is None else k_min)
    hi_cfg = int(CONFIG.cluster_k_max if k_max is None else k_max)
    n_init = int(CONFIG.kmeans_n_init if kmeans_n_init is None else kmeans_n_init)
    rs = int(CONFIG.kmeans_random_seed if random_state is None else random_state)

    w0 = returns.index.min()
    w1 = returns.index.max()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns.T)
    n_stocks = scaled.shape[0]

    if n_stocks < 2:
        raise ValueError("Need at least two tickers to cluster.")

    chosen_k: int
    sil_at_chosen: float | None = None

    if use_sil:
        lo_eff = max(2, lo)
        hi_eff = min(hi_cfg, n_stocks - 1)
        best_sil = -np.inf
        best_labels: np.ndarray | None = None
        best_k_run: int | None = None
        if hi_eff >= lo_eff:
            for k in range(lo_eff, hi_eff + 1):
                km = KMeans(n_clusters=k, random_state=rs, n_init=n_init)
                labels = km.fit_predict(scaled)
                if len(np.unique(labels)) < 2:
                    continue
                try:
                    sil = float(silhouette_score(scaled, labels, metric="euclidean"))
                except ValueError:
                    continue
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels.copy()
                    best_k_run = k
        if best_labels is not None and best_k_run is not None:
            chosen_k = best_k_run
            sil_at_chosen = float(best_sil)
            labels_final = best_labels
        else:
            chosen_k = max(2, min(n_fallback, n_stocks - 1))
            km_fb = KMeans(n_clusters=chosen_k, random_state=rs, n_init=n_init)
            labels_final = km_fb.fit_predict(scaled)
            try:
                sil_at_chosen = float(
                    silhouette_score(scaled, labels_final, metric="euclidean")
                ) if len(np.unique(labels_final)) >= 2 else None
            except ValueError:
                sil_at_chosen = None
            if verbose:
                print(
                    f"  Silhouette k-search failed or empty; fallback k={chosen_k} "
                    f"(silhouette={sil_at_chosen})."
                )
    else:
        chosen_k = max(2, min(n_fallback, n_stocks - 1))
        km = KMeans(n_clusters=chosen_k, random_state=rs, n_init=n_init)
        labels_final = km.fit_predict(scaled)
        sil_at_chosen = None
        if len(np.unique(labels_final)) >= 2:
            try:
                sil_at_chosen = float(silhouette_score(scaled, labels_final, metric="euclidean"))
            except ValueError:
                sil_at_chosen = None

    if verbose:
        sil_str = f"{sil_at_chosen:.4f}" if sil_at_chosen is not None else "n/a"
        mode = "silhouette-max" if use_sil else "fixed-k"
        print(
            f"Clustering {returns.shape[1]} stocks into k={chosen_k} clusters ({mode}, "
            f"silhouette={sil_str}) | returns {pd.Timestamp(w0).date()} .. "
            f"{pd.Timestamp(w1).date()} ({len(returns)} rows)"
        )

    return pd.Series(labels_final, index=returns.columns)


# --- Phase 3: Cointegration Testing ---
def find_cointegrated_pairs(prices, clusters, min_history=126):
    """Run Engle-Granger tests within each cluster on the provided price slice only.

    Args:
        prices: Wide price panel (typically a formation window only).
        clusters: Series ticker -> cluster id from cluster_stocks on the same window.
        min_history: Minimum overlapping days required to test a pair.

    Returns:
        DataFrame of passing pairs sorted by p-value.
    """
    print("\nTesting for cointegration within clusters...")
    print(
        f"  Price panel for tests: {pd.Timestamp(prices.index.min()).date()} .. "
        f"{pd.Timestamp(prices.index.max()).date()} ({len(prices)} rows)"
    )
    cointegrated_pairs = []
    
    # Iterate through each unique cluster
    for cluster_id in clusters.unique():
        cluster_members = clusters[clusters == cluster_id].index.tolist()
        
        # We need at least 2 stocks to form a pair
        if len(cluster_members) < 2:
            continue
            
        # Generate all possible pairs in this cluster
        pairs = list(itertools.combinations(cluster_members, 2))
        
        for stock_a, stock_b in pairs:
            # Extract price series
            series_a = prices[stock_a].dropna()
            series_b = prices[stock_b].dropna()
            
            # Align dates just in case
            common_dates = series_a.index.intersection(series_b.index)
            series_a = series_a.loc[common_dates]
            series_b = series_b.loc[common_dates]
            
            if len(series_a) < min_history:
                continue
                
            # Perform cointegration test (Engle-Granger)
            # Null hypothesis is that they are NOT cointegrated.
            score, pvalue, _ = ts.coint(series_a, series_b)
            
            # If p-value < 0.05, we reject the null and consider the pair cointegrated
            if pvalue < 0.05:
                cointegrated_pairs.append({
                    'Stock_A': stock_a,
                    'Stock_B': stock_b,
                    'Cluster': cluster_id,
                    'p_value': pvalue
                })
                
    result_df = pd.DataFrame(cointegrated_pairs)
    if not result_df.empty:
        result_df = result_df.sort_values(by='p_value').reset_index(drop=True)
    return result_df

# --- Phase 4: Spread Modeling & Half-Life Calculation ---
def calculate_half_life(spread):
    """Estimate mean-reversion half-life (trading days) from an AR(1) on the spread.

    Args:
        spread: Price spread series.

    Returns:
        Half-life in days, or inf if not mean-reverting.
    """
    df_spread = pd.DataFrame({'spread': spread})
    df_spread['spread_lag'] = df_spread['spread'].shift(1)
    df_spread['spread_diff'] = df_spread['spread'] - df_spread['spread_lag']
    df_spread = df_spread.dropna()
    
    # Linear regression: dz = lambda * z_lag + const
    X = df_spread['spread_lag']
    X = sm.add_constant(X)
    Y = df_spread['spread_diff']
    
    model = sm.OLS(Y, X).fit()
    lam = model.params.iloc[1] # The lambda value (coefficient for spread_lag)
    
    # If lambda >= 0, the series is not mean-reverting
    if lam >= 0:
        return np.inf 
        
    half_life = -np.log(2) / lam
    return half_life

def process_pairs_half_life(prices, pairs_df):
    """Estimate OLS hedge ratio and half-life per pair on the given price slice only.

    Args:
        prices: Wide price panel (formation window).
        pairs_df: DataFrame with Stock_A, Stock_B, Cluster, p_value.

    Returns:
        DataFrame with hedge ratio and half-life, filtered by config min/max half-life.
    """
    print("\nCalculating half-life for cointegrated pairs...")
    print(
        f"  Price panel: {pd.Timestamp(prices.index.min()).date()} .. "
        f"{pd.Timestamp(prices.index.max()).date()} ({len(prices)} rows)"
    )
    print(
        f"  Tradable half-life range: {CONFIG.half_life_min_days} .. "
        f"{CONFIG.half_life_max_days} days"
    )
    results = []
    
    for _, row in pairs_df.iterrows():
        stock_a = row['Stock_A']
        stock_b = row['Stock_B']
        
        series_a = prices[stock_a].dropna()
        series_b = prices[stock_b].dropna()
        
        # Align dates
        common_dates = series_a.index.intersection(series_b.index)
        series_a = series_a.loc[common_dates]
        series_b = series_b.loc[common_dates]
        
        # Calculate Hedge Ratio using OLS
        X = series_b
        X = sm.add_constant(X)
        Y = series_a
        
        model = sm.OLS(Y, X).fit()
        hedge_ratio = model.params.iloc[1] # Beta coefficient
        
        # Calculate the spread
        spread = series_a - (hedge_ratio * series_b)
        
        # Calculate half-life
        half_life = calculate_half_life(spread)
        
        results.append({
            'Stock_A': stock_a,
            'Stock_B': stock_b,
            'Cluster': row['Cluster'],
            'p_value': row['p_value'],
            'Hedge_Ratio': hedge_ratio,
            'Half_Life_Days': half_life
        })
        
    results_df = pd.DataFrame(results)
    # Filter for tradable half-lives using dedicated config bounds.
    tradable_pairs = results_df[
        (results_df['Half_Life_Days'] >= CONFIG.half_life_min_days)
        & (results_df['Half_Life_Days'] <= CONFIG.half_life_max_days)
    ]
    
    return tradable_pairs.sort_values(by='Half_Life_Days').reset_index(drop=True)

# --- Phase 5: Signal Generation & Backtesting ---
# (Moved to backtester.py)

def plot_pair(prices, stock_a, stock_b, hedge_ratio, half_life, entry_z=2.0, exit_z=0.0):
    """Plot normalized prices, lagged rolling z-score, and cumulative strategy returns.

    Args:
        prices: Wide price panel.
        stock_a: First ticker.
        stock_b: Second ticker.
        hedge_ratio: OLS hedge from a formation window (caller must avoid leakage).
        half_life: Half-life in days used for z-score window sizing.
        entry_z: Entry threshold on lagged z.
        exit_z: Exit threshold on lagged z.
    """
    from backtester import lagged_z_score

    print(f"\nGenerating plots for {stock_a} and {stock_b}...")
    
    # Align dates
    series_a = prices[stock_a].dropna()
    series_b = prices[stock_b].dropna()
    common_dates = series_a.index.intersection(series_b.index)
    series_a = series_a.loc[common_dates]
    series_b = series_b.loc[common_dates]
    
    # Calculate Spread
    spread = series_a - (hedge_ratio * series_b)
    
    window = max(5, int(min(half_life, 252)))
    z_score = lagged_z_score(spread, window)

    signals = pd.Series(0, index=z_score.index)
    position = 0
    for i in range(1, len(z_score)):
        z = z_score.iloc[i]
        if pd.isna(z):
            continue
        if position == 1 and z >= exit_z:
            position = 0
        elif position == -1 and z <= exit_z:
            position = 0
        if position == 0:
            if z < -entry_z:
                position = 1
            elif z > entry_z:
                position = -1
        signals.iloc[i] = position

    ret_a = series_a.pct_change()
    ret_b = series_b.pct_change()
    weight_a = 1 / (1 + abs(hedge_ratio))
    weight_b = -hedge_ratio / (1 + abs(hedge_ratio))
    strategy_returns = signals.shift(1) * (weight_a * ret_a + weight_b * ret_b)
    cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Plot 1: Normalized Prices
    (series_a / series_a.iloc[0]).plot(ax=axes[0], label=f'{stock_a} (Normalized)')
    (series_b / series_b.iloc[0]).plot(ax=axes[0], label=f'{stock_b} (Normalized)')
    axes[0].set_title(f"Normalized Price Series: {stock_a} vs {stock_b}")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Z-Score and Signals
    z_score.plot(ax=axes[1], color='blue', label='Z-Score')
    axes[1].axhline(y=entry_z, color='red', linestyle='--', label=f'Short Entry (+{entry_z})')
    axes[1].axhline(y=-entry_z, color='green', linestyle='--', label=f'Long Entry (-{entry_z})')
    axes[1].axhline(y=exit_z, color='black', linestyle=':', label='Exit (0.0)')
    axes[1].set_title("Rolling Z-Score & Trading Signals")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Cumulative Returns
    cumulative_returns.plot(ax=axes[2], color='purple', label='Cumulative Return')
    axes[2].set_title("Strategy Cumulative Returns")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    filename = f"{stock_a}_{stock_b}_pairs_trade.png"
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    import logging

    from backtester import summarize_and_plot, walk_forward_backtest

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Working-model pairs walk-forward backtest (optional price cache)."
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch from yfinance and overwrite parquet in cache/raw.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache raw directory (default: working_model/cache/raw).",
    )
    parser.add_argument(
        "--use-start",
        type=str,
        default=None,
        help="Inclusive first date for the working price panel (subset rows after parquet/yfinance load).",
    )
    parser.add_argument(
        "--use-end",
        type=str,
        default=None,
        help="Inclusive last date for the working price panel (subset after load).",
    )
    parser.add_argument(
        "--use-vix-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.use_vix_filter: --use-vix-filter / --no-use-vix-filter "
        "(omit to use configuration.py).",
    )
    parser.add_argument(
        "--use-silhouette-k-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CONFIG.use_silhouette_k_selection: pick KMeans k via silhouette, "
        "or use fixed n_clusters (default 5) with --no-use-silhouette-k-selection "
        "(omit to use configuration.py).",
    )
    args = parser.parse_args()
    cache_raw_dir = Path(args.cache_dir).resolve() if args.cache_dir else default_cache_raw_dir()

    example_tickers = CONFIG.tickers
    fetch_tickers = sorted(set(example_tickers + [CONFIG.spy_ticker]))
    start_date = CONFIG.fetch_start_date
    end_date = CONFIG.fetch_end_date
    panel_use_start = args.use_start if args.use_start is not None else CONFIG.panel_use_start
    panel_use_end = args.use_end if args.use_end is not None else CONFIG.panel_use_end
    backtest_start = CONFIG.backtest_start
    backtest_end = CONFIG.backtest_end
    formation_days = CONFIG.formation_days
    rescore_freq = CONFIG.rescore_freq_trading_days
    n_clusters = CONFIG.n_clusters
    min_coint_history = CONFIG.min_coint_history
    use_vix_filter = (
        CONFIG.use_vix_filter if args.use_vix_filter is None else args.use_vix_filter
    )
    use_silhouette_k_selection = (
        CONFIG.use_silhouette_k_selection
        if args.use_silhouette_k_selection is None
        else args.use_silhouette_k_selection
    )

    prices, price_meta = load_or_fetch_prices(
        fetch_tickers,
        start_date,
        end_date,
        cache_raw_dir=cache_raw_dir,
        force_refresh=args.force_refresh,
        use_start=panel_use_start,
        use_end=panel_use_end,
    )
    print(format_price_load_report(price_meta))
    volumes, _ = load_or_fetch_volumes(
        fetch_tickers,
        start_date,
        end_date,
        cache_raw_dir=cache_raw_dir,
        force_refresh=args.force_refresh,
        use_start=panel_use_start,
        use_end=panel_use_end,
    )
    returns = calculate_returns(prices)

    vix_series: pd.Series | None = None
    if use_vix_filter:
        vix_series, vix_meta = load_or_fetch_vix(
            start_date,
            end_date,
            vix_ticker=CONFIG.vix_ticker,
            cache_raw_dir=cache_raw_dir,
            force_refresh=args.force_refresh,
            use_start=panel_use_start,
            use_end=panel_use_end,
        )
        print(format_vix_load_report(vix_meta))

    # Point-in-time snapshot (last formation window strictly before OOS) for console only.
    bt0 = pd.Timestamp(backtest_start)
    pre = prices.loc[prices.index < bt0]
    if len(pre) >= formation_days:
        snap = pre.iloc[-formation_days:]
        snap_rets = calculate_returns(snap)
        min_snap = CONFIG.cluster_k_min if use_silhouette_k_selection else n_clusters
        if snap_rets.shape[0] >= 20 and snap_rets.shape[1] >= min_snap:
            print(
                "\n--- Example cluster snapshot (formation prices "
                f"{snap.index.min().date()} .. {snap.index.max().date()}, "
                f"{len(snap)} rows; returns used in clustering line below) ---"
            )
            snap_clusters = cluster_stocks(
                snap_rets,
                n_clusters=n_clusters,
                use_silhouette_k_selection=use_silhouette_k_selection,
                k_min=CONFIG.cluster_k_min,
                k_max=CONFIG.cluster_k_max,
                kmeans_n_init=CONFIG.kmeans_n_init,
                random_state=CONFIG.kmeans_random_seed,
                verbose=True,
            )
            print("--- Cluster labels (same returns window as clustering line above) ---")
            for cluster_id in sorted(snap_clusters.unique()):
                members = snap_clusters[snap_clusters == cluster_id].index.tolist()
                print(f"\nCluster {int(cluster_id)} (Size: {len(members)}): {', '.join(members)}")

    print("\n--- Walk-forward OOS backtest (no look-ahead) ---")
    portfolio_returns, nav_series, active_pairs_series, pair_pnl_df, pair_stats, _ = walk_forward_backtest(
        prices,
        volumes,
        returns,
        tradable_tickers=example_tickers,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        formation_days=formation_days,
        rescore_freq_trading_days=rescore_freq,
        n_clusters=n_clusters,
        min_coint_history=min_coint_history,
        entry_z=CONFIG.entry_z,
        exit_z=CONFIG.exit_z,
        stop_loss_z=CONFIG.stop_loss_z,
        max_holding_multiplier=CONFIG.max_holding_multiplier,
        initial_capital=CONFIG.initial_capital,
        max_active_pairs=CONFIG.max_active_pairs,
        target_gross_per_pair_pct=CONFIG.target_gross_per_pair_pct,
        max_gross_exposure_pct=CONFIG.max_gross_exposure_pct,
        commission_bps=CONFIG.commission_bps,
        slippage_bps=CONFIG.slippage_bps,
        short_borrow_annual=CONFIG.short_borrow_annual,
        trading_days_per_year=CONFIG.trading_days_per_year,
        vix=vix_series,
        use_vix_filter=use_vix_filter,
        vix_entry_block=CONFIG.vix_entry_block,
        vix_resume=CONFIG.vix_resume,
        vix_resume_days=CONFIG.vix_resume_days,
        use_silhouette_k_selection=use_silhouette_k_selection,
        verbose=True,
    )
    summarize_and_plot(
        portfolio_returns,
        nav_series,
        active_pairs_series,
        pair_pnl_df,
        pair_stats,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        training_start=price_meta.get("panel_first"),
        timeline_plot_path="portfolio_timeline.png",
    )

    print("\nPairs trading walk-forward run complete.")
