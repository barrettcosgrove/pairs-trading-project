import numpy as np
import pandas as pd

# Calendar start of the historical 30% tail (2024-02-08). This is a reporting
# cut of script 03 CSVs, not a walk-forward run.
OOS_START = "2024-02-08"


def analyze_oos():
    """Print 2024-tail stats from the full backtest CSVs."""
    try:
        tl = pd.read_csv("outputs/backtest_results/trade_log.csv")
        nav = pd.read_csv("outputs/backtest_results/nav_series.csv")
    except FileNotFoundError:
        print("Full backtest CSVs not found. Please run scripts/03_run_backtest.py first.")
        return

    oos_start = pd.Timestamp(OOS_START)
    if "date" not in tl.columns or "date" not in nav.columns:
        print("trade_log/nav_series missing required column 'date'.")
        return

    tl["date"] = pd.to_datetime(tl["date"])
    nav["date"] = pd.to_datetime(nav["date"])
    tl = tl.loc[tl["date"] >= oos_start].copy()
    nav = nav.loc[nav["date"] >= oos_start].copy()

    print("=" * 50)
    print(f"  TAIL SLICE ANALYSIS (date >= {OOS_START})")
    print("=" * 50)
    print(f"Total trade rows: {len(tl)}")

    if not tl.empty:
        entries = tl[tl["action"].isin(["LONG_SPREAD", "SHORT_SPREAD"])]
        exits = tl[tl["action"].isin([
            "TAKE_PROFIT", "STOP_LOSS", "TIME_STOP",
            "PLATEAU_STOP", "EARNINGS_EXIT", "DOLLAR_STOP",
        ])]

        print(f"Entries: {len(entries)}, Exits: {len(exits)}")
        print("\nAction counts:")
        print(tl["action"].value_counts().to_string())
        print(f"\nAverage Cost per trade: ${tl['cost'].mean():.2f}")

        if not exits.empty and "pnl" in exits.columns:
            wins = exits[exits["pnl"] > 0]
            win_rate = len(wins) / len(exits) * 100
            print(f"Win rate: {win_rate:.2f}%")

    if not nav.empty:
        start_nav = nav["nav"].iloc[0]
        end_nav = nav["nav"].iloc[-1]
        ret = (end_nav - start_nav) / start_nav * 100 if start_nav else 0.0

        max_dd = nav["drawdown_from_peak"].max() * 100

        daily_returns = nav["nav"].pct_change().dropna()
        if not daily_returns.empty and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        print("\n" + "=" * 50)
        print("  TAIL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Start NAV    : ${start_nav:,.2f}")
        print(f"End NAV      : ${end_nav:,.2f}")
        print(f"Total Return : {ret:+.2f}%")
        print(f"Max Drawdown : {max_dd:.2f}%")
        print(f"Sharpe Ratio : {sharpe_ratio:.2f}")


if __name__ == "__main__":
    analyze_oos()
