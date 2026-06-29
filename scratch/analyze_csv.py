import numpy as np
import pandas as pd

tl = pd.read_csv('outputs/backtest_results/trade_log.csv')
nav = pd.read_csv('outputs/backtest_results/nav_series.csv')

print(f"Total Trades: {len(tl)}")

entries = tl[tl['action'].isin(['LONG_SPREAD', 'SHORT_SPREAD'])]
exits = tl[tl['action'].isin(['TAKE_PROFIT', 'STOP_LOSS', 'TIME_STOP'])]

print(f"Entries: {len(entries)}, Exits: {len(exits)}")
print("\nAction counts:")
print(tl['action'].value_counts())

print("\nNAV Summary:")
print(nav[['nav', 'cash', 'gross_exposure', 'drawdown_from_peak']].describe())

if not nav.empty:
    start_nav = nav['nav'].iloc[0]
    end_nav = nav['nav'].iloc[-1]
    ret = (end_nav - start_nav) / start_nav * 100
    print(f"\nStart NAV: {start_nav:.2f}, End NAV: {end_nav:.2f}")
    print(f"Total Return: {ret:.2f}%")

    # Calculate Annualized Sharpe Ratio
    daily_returns = nav['nav'].pct_change().dropna()
    risk_free_daily = 0.04 / 252  # Assuming 4% risk-free rate
    excess_returns = daily_returns - risk_free_daily
    if not excess_returns.empty and excess_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

print(f"Average Cost per trade: {tl['cost'].mean():.2f}")

if not exits.empty:
    wins = exits[exits['pnl'] > 0]
    win_rate = len(wins) / len(exits) * 100
    print(f"Win rate: {win_rate:.2f}%")
