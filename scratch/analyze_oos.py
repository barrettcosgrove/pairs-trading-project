import numpy as np
import pandas as pd

def analyze_oos():
    try:
        tl = pd.read_csv('outputs/backtest_results/oos_trade_log.csv')
        nav = pd.read_csv('outputs/backtest_results/oos_nav_series.csv')
    except FileNotFoundError:
        print("OOS files not found. Please run scripts/04_walkforward.py first.")
        return
    
    print("=" * 50)
    print("  OUT-OF-SAMPLE (OOS) ANALYSIS")
    print("=" * 50)
    print(f"Total OOS Trades: {len(tl)}")
    
    if not tl.empty:
        entries = tl[tl['action'].isin(['LONG_SPREAD', 'SHORT_SPREAD'])]
        exits = tl[tl['action'].isin(['TAKE_PROFIT', 'STOP_LOSS', 'TIME_STOP'])]
        
        print(f"Entries: {len(entries)}, Exits: {len(exits)}")
        print("\nAction counts:")
        print(tl['action'].value_counts().to_string())
        print(f"\nAverage Cost per trade: ${tl['cost'].mean():.2f}")
        
        if not exits.empty:
            wins = exits[exits['pnl'] > 0]
            win_rate = len(wins) / len(exits) * 100
            print(f"Win rate: {win_rate:.2f}%")
            
    if not nav.empty:
        start_nav = nav['nav'].iloc[0]
        end_nav = nav['nav'].iloc[-1]
        ret = (end_nav - start_nav) / start_nav * 100
        
        # Max Drawdown
        max_dd = nav['drawdown_from_peak'].max() * 100
        
        # Annualized Sharpe Ratio
        daily_returns = nav['nav'].pct_change().dropna()
        risk_free_daily = 0.04 / 252  # 4% annual risk-free rate
        excess_returns = daily_returns - risk_free_daily
        
        if not excess_returns.empty and excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
            
        print("\n" + "=" * 50)
        print("  OOS PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Start NAV    : ${start_nav:,.2f}")
        print(f"End NAV      : ${end_nav:,.2f}")
        print(f"Total Return : {ret:+.2f}%")
        print(f"Max Drawdown : {max_dd:.2f}%")
        print(f"Sharpe Ratio : {sharpe_ratio:.2f}")

if __name__ == "__main__":
    analyze_oos()
