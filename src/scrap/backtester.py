"""Walk-forward portfolio simulation for scrap pairs research (no look-ahead)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pairs_strategy import (
    calculate_returns,
    cluster_stocks,
    find_cointegrated_pairs,
    process_pairs_half_life,
)


class PairsRiskManager:
    """Simple per-pair risk checks (z stop and time stop vs half-life)."""

    def __init__(self, stop_loss_z: float = 4.0, max_holding_multiplier: float = 3.0) -> None:
        self.stop_loss_z = stop_loss_z
        self.max_holding_multiplier = max_holding_multiplier

    def check_stop_loss(self, z_score: float) -> bool:
        """Return True if absolute z exceeds the stop threshold."""
        return abs(z_score) >= self.stop_loss_z

    def check_time_stop(self, days_held: int, half_life: float) -> bool:
        """Return True if held longer than max_holding_multiplier * half_life."""
        return days_held > (half_life * self.max_holding_multiplier)


def lagged_z_score(spread: pd.Series, window: int) -> pd.Series:
    """Rolling z using only information through t-1 for the value at time t.

    At index t, uses spread.shift(1) vs rolling mean/std computed with .shift(1)
    so mean and std at t do not include spread[t].

    Args:
        spread: Price spread series indexed by date.
        window: Rolling window length (trading days).

    Returns:
        Z-score series aligned to spread index (NaN until enough history).
    """
    w = max(5, int(window))
    sp_lag = spread.shift(1)
    mu = spread.rolling(window=w, min_periods=w).mean().shift(1)
    sd = spread.rolling(window=w, min_periods=w).std().shift(1)
    return (sp_lag - mu) / sd.clip(lower=1e-9)


def _z_window_from_half_life(half_life: float) -> int:
    """Map half-life (days) to rolling z window with sane bounds."""
    if half_life is None or not np.isfinite(half_life) or half_life <= 0:
        return 5
    return max(5, int(min(half_life, 252)))


def walk_forward_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    backtest_start: str,
    backtest_end: str,
    formation_days: int = 252,
    rescore_freq_trading_days: int = 21,
    n_clusters: int = 5,
    min_coint_history: int = 200,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_loss_z: float = 4.0,
    max_holding_multiplier: float = 3.0,
    verbose: bool = True,
) -> tuple[pd.Series, list[dict], pd.DatetimeIndex]:
    """Run a walk-forward backtest with no in-sample leakage into OOS decisions.

    Discovery (cluster, Engle-Granger, OLS hedge, half-life) uses only prices
    with index strictly before each segment's first live day. Signals use
    lagged_z_score so decisions at t depend only on data through t-1.

    Args:
        prices: Wide adjusted close panel, DatetimeIndex rows.
        returns: Simple pct-change returns aligned to prices (may start one row later).
        backtest_start: First date (inclusive) to include in performance stats.
        backtest_end: Last date (inclusive) for performance stats.
        formation_days: Trading days of history before each segment's first live day.
        rescore_freq_trading_days: Segment length; re-run discovery each segment.
        n_clusters: KMeans cluster count.
        min_coint_history: Minimum overlapping days required for cointegration test.
        entry_z: Entry threshold on lagged z.
        exit_z: Exit threshold on lagged z.
        stop_loss_z: Stop when |lagged z| exceeds this.
        max_holding_multiplier: Time stop when days_held exceeds this times half-life.
        verbose: If True, print each segment's formation vs live date ranges before clustering.

    Returns:
        Tuple of (portfolio daily returns for OOS dates only, per-pair stats list,
        DatetimeIndex of OOS dates included in portfolio_returns).
    """
    risk_manager = PairsRiskManager(
        stop_loss_z=stop_loss_z,
        max_holding_multiplier=max_holding_multiplier,
    )

    cal = prices.index.sort_values()
    bt_start = pd.Timestamp(backtest_start)
    bt_end = pd.Timestamp(backtest_end)

    first_bt_pos = None
    for i, d in enumerate(cal):
        if d < bt_start:
            continue
        if i < formation_days:
            continue
        first_bt_pos = i
        break
    if first_bt_pos is None:
        raise ValueError(
            "No valid OOS start: need enough history before backtest_start "
            f"(formation_days={formation_days})."
        )

    last_bt_pos = int(np.where(cal <= bt_end)[0].max())

    oos_returns: dict[pd.Timestamp, float] = {}
    pair_stats_map: dict[str, dict[str, int]] = {}

    seg_start = first_bt_pos
    while seg_start <= last_bt_pos:
        seg_end = min(seg_start + rescore_freq_trading_days - 1, last_bt_pos)
        form_lo = seg_start - formation_days
        form_hi = seg_start  # exclusive: rows form_lo .. seg_start-1
        formation_prices = prices.iloc[form_lo:form_hi]
        formation_returns = calculate_returns(formation_prices)

        if formation_returns.shape[0] < 20 or formation_returns.shape[1] < n_clusters:
            seg_start = seg_end + 1
            continue

        if verbose:
            fp0, fp1 = cal[form_lo], cal[seg_start - 1]
            lp0, lp1 = cal[seg_start], cal[seg_end]
            print(
                f"\n--- Walk-forward segment | formation prices: {fp0.date()} .. {fp1.date()} "
                f"({formation_days} trading rows) | live: {lp0.date()} .. {lp1.date()} "
                f"({seg_end - seg_start + 1} days) ---"
            )

        clusters = cluster_stocks(formation_returns, n_clusters=n_clusters)
        coint = find_cointegrated_pairs(
            formation_prices,
            clusters,
            min_history=min(min_coint_history, len(formation_prices) - 5),
        )
        if coint.empty:
            seg_start = seg_end + 1
            continue

        tradable = process_pairs_half_life(formation_prices, coint)
        if tradable.empty:
            seg_start = seg_end + 1
            continue

        segment_cal = cal[seg_start : seg_end + 1]
        day_contribs: dict[pd.Timestamp, list[float]] = {d: [] for d in segment_cal}

        for _, row in tradable.iterrows():
            stock_a = row["Stock_A"]
            stock_b = row["Stock_B"]
            hedge_ratio = float(row["Hedge_Ratio"])
            half_life = float(row["Half_Life_Days"])
            pair_key = f"{stock_a}_{stock_b}"

            if pair_key not in pair_stats_map:
                pair_stats_map[pair_key] = {"Trades_Won": 0, "Trades_Lost": 0}

            if stock_a not in prices.columns or stock_b not in prices.columns:
                continue

            series_a = prices[stock_a].dropna()
            series_b = prices[stock_b].dropna()
            common = series_a.index.intersection(series_b.index)
            series_a = series_a.loc[common]
            series_b = series_b.loc[common]

            # Spread path uses hedge frozen from formation; include formation history
            # so rolling z at first OOS day uses only past spreads (no future OOS).
            hist_start = cal[form_lo]
            hist_end = cal[seg_end]
            sa = series_a.loc[hist_start:hist_end]
            sb = series_b.loc[hist_start:hist_end]
            spread = sa - hedge_ratio * sb
            zw = _z_window_from_half_life(half_life)
            z_signal = lagged_z_score(spread, zw)

            ret_a = sa.pct_change()
            ret_b = sb.pct_change()
            weight_a = 1.0 / (1.0 + abs(hedge_ratio))
            weight_b = -hedge_ratio / (1.0 + abs(hedge_ratio))

            position = 0
            days_held = 0
            prev_signal = 0

            for d in segment_cal:
                leg_r = weight_a * ret_a.loc[d] + weight_b * ret_b.loc[d]
                strat_r = prev_signal * leg_r
                if not pd.isna(strat_r):
                    day_contribs[d].append(float(strat_r))

                z = z_signal.loc[d] if d in z_signal.index else np.nan
                if pd.isna(z):
                    prev_signal = position
                    continue

                if position != 0:
                    days_held += 1
                    if risk_manager.check_stop_loss(float(z)):
                        position = 0
                        pair_stats_map[pair_key]["Trades_Lost"] += 1
                    elif risk_manager.check_time_stop(days_held, half_life):
                        position = 0
                        pair_stats_map[pair_key]["Trades_Lost"] += 1
                    elif (position == 1 and z >= exit_z) or (position == -1 and z <= exit_z):
                        position = 0
                        pair_stats_map[pair_key]["Trades_Won"] += 1

                if position == 0:
                    days_held = 0

                if position == 0:
                    if z < -entry_z:
                        position = 1
                    elif z > entry_z:
                        position = -1

                prev_signal = position

            # Flatten at segment end: next rescore uses a new hedge; do not carry exposure.
            if position != 0:
                position = 0
                days_held = 0
                prev_signal = 0

        for d in segment_cal:
            contribs = day_contribs.get(d, [])
            if not contribs:
                continue
            oos_returns[d] = float(np.mean(contribs))

        seg_start = seg_end + 1

    oos_index = cal[(cal >= bt_start) & (cal <= bt_end)]
    portfolio_returns = pd.Series(0.0, index=oos_index, dtype=float)
    for d, r in oos_returns.items():
        if d in portfolio_returns.index:
            portfolio_returns.loc[d] = r

    pair_stats = [
        {"Pair": k, "Trades_Won": v["Trades_Won"], "Trades_Lost": v["Trades_Lost"]}
        for k, v in sorted(pair_stats_map.items())
    ]

    return portfolio_returns, pair_stats, oos_index


def summarize_and_plot(
    portfolio_returns: pd.Series,
    pair_stats: list[dict],
    *,
    backtest_start: str,
    backtest_end: str,
    plot_path: str = "portfolio_performance.png",
) -> None:
    """Print summary metrics and save cumulative return plot.

    Args:
        portfolio_returns: Daily OOS returns.
        pair_stats: Per-pair win/loss counts from walk_forward_backtest.
        backtest_start: Label for printout.
        backtest_end: Label for printout.
        plot_path: Output path for PNG.
    """
    cumulative_returns = (1 + portfolio_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) else 0.0

    excess_returns = portfolio_returns
    if excess_returns.std() > 0:
        sharpe_ratio = float(np.sqrt(252) * (excess_returns.mean() / excess_returns.std()))
    else:
        sharpe_ratio = 0.0

    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    total_won = sum(p["Trades_Won"] for p in pair_stats)
    total_lost = sum(p["Trades_Lost"] for p in pair_stats)
    total_trades = total_won + total_lost
    win_rate = total_won / total_trades if total_trades > 0 else 0.0

    print("\n=========================================")
    print("   WALK-FORWARD PORTFOLIO (NO LOOK-AHEAD)")
    print("=========================================")
    print(f"OOS window:         {backtest_start} .. {backtest_end}")
    print(f"Total Return:       {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate * 100:.2f}%")
    print("=========================================")

    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(color="teal", label="OOS cumulative return", linewidth=2)
    plt.title("Walk-forward pairs portfolio (out-of-sample)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Portfolio performance graph saved to {plot_path!r}")


def backtest_portfolio(prices, tradable_pairs, entry_z=2.0, exit_z=0.0):
    """Legacy full-sample backtest (in-sample leakage). Prefer walk_forward_backtest.

    Args:
        prices: Wide price panel.
        tradable_pairs: DataFrame with Stock_A, Stock_B, Hedge_Ratio, Half_Life_Days.
        entry_z: Entry z threshold.
        exit_z: Exit z threshold.

    Returns:
        (portfolio_returns, pair_stats) as before.
    """
    print("\nStarting Portfolio Backtest (legacy in-sample; not walk-forward)...")

    risk_manager = PairsRiskManager(stop_loss_z=4.0, max_holding_multiplier=3.0)

    all_pair_returns = pd.DataFrame(index=prices.index)
    pair_stats = []

    for _, row in tradable_pairs.iterrows():
        stock_a = row["Stock_A"]
        stock_b = row["Stock_B"]
        hedge_ratio = row["Hedge_Ratio"]
        half_life = row["Half_Life_Days"]

        series_a = prices[stock_a].dropna()
        series_b = prices[stock_b].dropna()
        common_dates = series_a.index.intersection(series_b.index)
        series_a = series_a.loc[common_dates]
        series_b = series_b.loc[common_dates]

        spread = series_a - (hedge_ratio * series_b)
        z_score = lagged_z_score(spread, _z_window_from_half_life(half_life))

        signals = pd.Series(0, index=z_score.index)
        position = 0
        days_held = 0
        trades_won = 0
        trades_lost = 0

        for i in range(1, len(z_score)):
            z = z_score.iloc[i]
            if pd.isna(z):
                continue

            if position != 0:
                days_held += 1

                if risk_manager.check_stop_loss(z):
                    position = 0
                    trades_lost += 1
                elif risk_manager.check_time_stop(days_held, half_life):
                    position = 0
                    trades_lost += 1
                elif (position == 1 and z >= exit_z) or (position == -1 and z <= exit_z):
                    position = 0
                    trades_won += 1

            if position == 0:
                days_held = 0

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
        pair_name = f"{stock_a}_{stock_b}"
        all_pair_returns[pair_name] = strategy_returns

        pair_stats.append({"Pair": pair_name, "Trades_Won": trades_won, "Trades_Lost": trades_lost})

    all_pair_returns = all_pair_returns.fillna(0)

    portfolio_returns = all_pair_returns.sum(axis=1) / max(len(tradable_pairs), 1)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1

    excess_returns = portfolio_returns
    if excess_returns.std() > 0:
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    else:
        sharpe_ratio = 0.0

    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1.0
    max_drawdown = drawdown.min()

    total_won = sum([p["Trades_Won"] for p in pair_stats])
    total_lost = sum([p["Trades_Lost"] for p in pair_stats])
    total_trades = total_won + total_lost
    win_rate = total_won / total_trades if total_trades > 0 else 0

    print("\n=========================================")
    print("      PORTFOLIO BACKTEST RESULTS         ")
    print("=========================================")
    print("Timeframe:          (full sample — legacy)")
    print(f"Total Return:       {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate * 100:.2f}%")
    print("=========================================")

    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(color="teal", label="Portfolio Cumulative Return", linewidth=2)
    plt.title("Pairs Trading Portfolio Performance (legacy)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("portfolio_performance.png")
    plt.close()
    print("Portfolio performance graph saved to 'portfolio_performance.png'")

    return portfolio_returns, pair_stats
