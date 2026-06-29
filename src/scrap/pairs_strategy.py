import argparse
import itertools
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt

from price_cache import (
    default_cache_raw_dir,
    format_price_load_report,
    load_or_fetch_prices,
)

def download_data(
    tickers,
    start_date,
    end_date,
    *,
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
):
    """Load adjusted close panel from cache or yfinance (same cleaning as ``load_or_fetch_prices``).

    Args:
        tickers: Iterable of ticker symbols.
        start_date: Inclusive start (string or datetime-like).
        end_date: Exclusive end for yfinance (string or datetime-like).
        cache_raw_dir: Optional directory for parquet cache (defaults to scrap ``cache/raw``).
        force_refresh: If True, refetch and overwrite cache.

    Returns:
        Wide DataFrame of closes indexed by date.
    """
    df, _meta = load_or_fetch_prices(
        list(tickers),
        str(start_date),
        str(end_date),
        cache_raw_dir=cache_raw_dir,
        force_refresh=force_refresh,
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
def cluster_stocks(returns, n_clusters=5):
    """Cluster stocks by standardized return vectors over the given return window.

    Args:
        returns: Wide simple returns; each column is a ticker.
        n_clusters: Number of KMeans clusters.

    Returns:
        Series mapping ticker to integer cluster id.
    """
    w0 = returns.index.min()
    w1 = returns.index.max()
    print(
        f"Clustering {returns.shape[1]} stocks into {n_clusters} clusters "
        f"(returns window: {pd.Timestamp(w0).date()} .. {pd.Timestamp(w1).date()}, "
        f"{len(returns)} rows)..."
    )
    
    # Transpose so that each row represents a stock and columns are the dates
    # We standardize the returns for each stock so variance doesn't dominate
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns.T) 
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_returns)
    
    # Create a Series mapping the ticker to its cluster ID
    clusters = pd.Series(kmeans.labels_, index=returns.columns)
    return clusters

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
        DataFrame with hedge ratio and half-life, filtered to 1..252 day half-life.
    """
    print("\nCalculating half-life for cointegrated pairs...")
    print(
        f"  Price panel: {pd.Timestamp(prices.index.min()).date()} .. "
        f"{pd.Timestamp(prices.index.max()).date()} ({len(prices)} rows)"
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
    # Filter for tradable half-lives (e.g., between 1 day and 1 year)
    tradable_pairs = results_df[(results_df['Half_Life_Days'] >= 1) & (results_df['Half_Life_Days'] <= 252)]
    
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

    parser = argparse.ArgumentParser(description="Scrap pairs walk-forward backtest (optional price cache).")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch from yfinance and overwrite parquet in cache/raw.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache raw directory (default: src/scrap/cache/raw).",
    )
    args = parser.parse_args()
    cache_raw_dir = Path(args.cache_dir).resolve() if args.cache_dir else default_cache_raw_dir()

    # Example Universe: A mix of Tech, Financials, Healthcare, and Energy stocks
    example_tickers = [
    # Semiconductors (8)
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT",
    
    # Cloud / Enterprise Software (10 - S&P 500 only)
    "MSFT", "CRM", "NOW", "ADSK", "CDNS", "SNPS", "WDAY", "ORCL", "ADBE", "ANET",
    
    # Cybersecurity (4 - the ones in S&P 500)
    "FTNT", "PANW", "CRWD", "CHKP",
    
    # Energy (8)
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO",
    
    # Financials (9)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW",
    
    # Healthcare (10)
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY", "BSX",
    
    # Consumer Staples (8)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "CL", "GIS",
    
    # Industrials (10)
    "CAT", "DE", "HON", "GE", "LMT", "UPS", "FDX", "EMR", "ETN", "RTX",
    
    # Utilities (8)
    "NEE", "DUK", "SO", "AEP", "EXC", "D", "XEL", "WEC",
    
    # Materials (6)
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX",
    
    # Consumer Discretionary (8)
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX",
    
    # Communication Services (6)
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T",
]
    
    # Full history for download; OOS backtest only uses [BACKTEST_START, BACKTEST_END].
    start_date = "2018-01-01"
    end_date = "2025-05-04"
    BACKTEST_START = "2023-01-01"
    BACKTEST_END = end_date
    FORMATION_DAYS = 252
    RESCORE_FREQ = 21
    N_CLUSTERS = 5
    MIN_COINT_HISTORY = 200

    prices, price_meta = load_or_fetch_prices(
        example_tickers,
        start_date,
        end_date,
        cache_raw_dir=cache_raw_dir,
        force_refresh=args.force_refresh,
    )
    print(format_price_load_report(price_meta))
    returns = calculate_returns(prices)

    # Point-in-time snapshot (last formation window strictly before OOS) for console only.
    bt0 = pd.Timestamp(BACKTEST_START)
    pre = prices.loc[prices.index < bt0]
    if len(pre) >= FORMATION_DAYS:
        snap = pre.iloc[-FORMATION_DAYS:]
        snap_rets = calculate_returns(snap)
        if snap_rets.shape[0] >= 20 and snap_rets.shape[1] >= N_CLUSTERS:
            print(
                "\n--- Example cluster snapshot (formation prices "
                f"{snap.index.min().date()} .. {snap.index.max().date()}, "
                f"{len(snap)} rows; returns used below) ---"
            )
            snap_clusters = cluster_stocks(snap_rets, n_clusters=N_CLUSTERS)
            print("--- Cluster labels (same returns window as clustering line above) ---")
            for cluster_id in range(N_CLUSTERS):
                members = snap_clusters[snap_clusters == cluster_id].index.tolist()
                print(f"\nCluster {cluster_id} (Size: {len(members)}): {', '.join(members)}")

    print("\n--- Walk-forward OOS backtest (no look-ahead) ---")
    portfolio_returns, pair_stats, _ = walk_forward_backtest(
        prices,
        returns,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        formation_days=FORMATION_DAYS,
        rescore_freq_trading_days=RESCORE_FREQ,
        n_clusters=N_CLUSTERS,
        min_coint_history=MIN_COINT_HISTORY,
        verbose=True,
    )
    summarize_and_plot(
        portfolio_returns,
        pair_stats,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
    )

    print("\nPairs trading walk-forward run complete.")
