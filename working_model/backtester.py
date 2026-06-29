"""Walk-forward backtest with true NAV accounting for working_model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

from configuration import CONFIG
from pairs_strategy import (
    calculate_returns,
    cluster_stocks,
    find_cointegrated_pairs,
    process_pairs_half_life,
)

logger = logging.getLogger(__name__)


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


def _align_vix_lagged_to_calendar(vix: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    """Reindex VIX to the equity calendar, fill gaps, shift(1) for no same-day peek.

    At equity date ``d`` the returned value is the VIX **close from the prior**
    date on ``cal`` (after ffill alignment). First row of ``cal`` is NaN by design.

    Args:
        vix: VIX close series with DatetimeIndex.
        cal: Sorted equity trading calendar (e.g. prices index).

    Returns:
        Lagged VIX series aligned to ``cal``.
    """
    v = pd.Series(
        np.asarray(vix, dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(vix.index)),
        name="VIX",
    )
    v = v.sort_index()
    v = v[~v.index.duplicated(keep="last")]
    aligned = v.reindex(cal).ffill().bfill()
    return aligned.shift(1)


def _step_vix_entry_permission(
    entries_ok: bool,
    streak: int,
    v_lag: float,
    *,
    entry_block: float,
    resume: float,
    resume_days: int,
) -> tuple[bool, int]:
    """Update hysteresis state: block new entries when lagged VIX is high.

    Uses lagged VIX only (see ``_align_vix_lagged_to_calendar``). When ``v_lag``
    is NaN, new entries are disallowed and streak resets (conservative).

    Args:
        entries_ok: Whether new entries were allowed after the previous day.
        streak: Consecutive days lagged VIX was at or below ``resume``.
        v_lag: Prior-trading-day VIX level for the current decision date.
        entry_block: Block new entries when ``v_lag >= entry_block``.
        resume: Count a calm day when ``v_lag <= resume``.
        resume_days: Consecutive calm days required to allow entries again.

    Returns:
        Updated ``(entries_ok, streak)``.
    """
    if pd.isna(v_lag):
        return False, 0
    if v_lag >= entry_block:
        return False, 0
    if v_lag <= resume:
        new_streak = streak + 1
        if new_streak >= resume_days:
            return True, new_streak
        return entries_ok, new_streak
    return entries_ok, 0


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


def _pair_gross_exposure(qty_a: float, price_a: float, qty_b: float, price_b: float) -> float:
    """Return pair gross dollar exposure."""
    return abs(qty_a * price_a) + abs(qty_b * price_b)


def _current_gross_exposure(positions: dict[str, dict], prices_t: pd.Series) -> float:
    """Compute portfolio gross exposure from currently open positions."""
    gross = 0.0
    for pos in positions.values():
        if not pos["is_open"]:
            continue
        a = pos["stock_a"]
        b = pos["stock_b"]
        if a not in prices_t.index or b not in prices_t.index:
            continue
        gross += _pair_gross_exposure(pos["qty_a"], float(prices_t[a]), pos["qty_b"], float(prices_t[b]))
    return gross


def _trade_cost(dollar_notional: float, commission_bps: float, slippage_bps: float) -> float:
    """Compute total transaction costs in dollars."""
    bps = (commission_bps + slippage_bps) / 10_000.0
    return dollar_notional * bps


def _target_qty_for_signal(
    signal: int,
    hedge_ratio: float,
    price_a: float,
    price_b: float,
    gross_target: float,
) -> tuple[float, float]:
    """Map spread signal into share quantities for stock A and B."""
    if signal == 0 or gross_target <= 0:
        return 0.0, 0.0
    denom = 1.0 + abs(hedge_ratio)
    dollar_a = signal * gross_target * (1.0 / denom)
    dollar_b = signal * gross_target * (-hedge_ratio / denom)
    qty_a = dollar_a / price_a
    qty_b = dollar_b / price_b
    return qty_a, qty_b


def _apply_target_position(
    pos: dict,
    target_qty_a: float,
    target_qty_b: float,
    price_a: float,
    price_b: float,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[float, float]:
    """Return (cash_delta, costs) when moving from current qty to target qty."""
    delta_a = target_qty_a - pos["qty_a"]
    delta_b = target_qty_b - pos["qty_b"]
    traded_notional = abs(delta_a * price_a) + abs(delta_b * price_b)
    costs = _trade_cost(traded_notional, commission_bps, slippage_bps)
    cash_delta = -(delta_a * price_a + delta_b * price_b) - costs
    return cash_delta, costs


def _mark_to_market(cash: float, positions: dict[str, dict], prices_t: pd.Series) -> float:
    """Compute end-of-day NAV from cash plus marked positions."""
    nav = cash
    for pos in positions.values():
        if not pos["is_open"]:
            continue
        a = pos["stock_a"]
        b = pos["stock_b"]
        if a not in prices_t.index or b not in prices_t.index:
            continue
        nav += pos["qty_a"] * float(prices_t[a]) + pos["qty_b"] * float(prices_t[b])
    return nav


def _passes_spy_correlation_gate(
    ticker: str,
    prices_to_asof: pd.DataFrame,
    *,
    spy_ticker: str,
    window: int,
    max_corr: float,
    min_obs: int,
) -> bool:
    """Return True if ticker's trailing correlation to SPY is below threshold."""
    if ticker == spy_ticker:
        return False
    if ticker not in prices_to_asof.columns or spy_ticker not in prices_to_asof.columns:
        return False
    ret_t = prices_to_asof[ticker].pct_change()
    ret_s = prices_to_asof[spy_ticker].pct_change()
    aligned = pd.concat([ret_t, ret_s], axis=1).dropna().tail(window)
    if len(aligned) < min_obs:
        return False
    corr = float(aligned.corr().iloc[0, 1])
    return corr < max_corr


def _filter_universe_for_segment(
    prices_to_asof: pd.DataFrame,
    volumes_to_asof: pd.DataFrame,
    *,
    tradable_tickers: list[str],
    min_price: float,
    min_adv: int,
    min_dollar_volume: float,
    liquidity_window: int,
    spy_ticker: str,
    spy_correlation_window: int,
    max_spy_correlation: float,
    spy_min_observations: int,
) -> tuple[list[str], dict[str, int]]:
    """Apply hard universe gates at segment formation end (point-in-time)."""
    keep: list[str] = []
    counts = {
        "base": 0,
        "missing_columns_or_history": 0,
        "price_floor": 0,
        "adv": 0,
        "dollar_volume": 0,
        "spy_correlation": 0,
        "kept": 0,
    }
    for t in tradable_tickers:
        counts["base"] += 1
        if t not in prices_to_asof.columns or t not in volumes_to_asof.columns:
            counts["missing_columns_or_history"] += 1
            continue
        px = prices_to_asof[t].dropna()
        vol = volumes_to_asof[t].dropna()
        if len(px) < liquidity_window or len(vol) < liquidity_window:
            counts["missing_columns_or_history"] += 1
            continue
        px_tail = px.tail(liquidity_window)
        vol_tail = vol.reindex(px_tail.index).dropna()
        if len(vol_tail) < liquidity_window:
            counts["missing_columns_or_history"] += 1
            continue
        latest_price = float(px_tail.iloc[-1])
        adv = float(vol_tail.mean())
        adv_dollar = float((px_tail * vol_tail).mean())
        if latest_price <= min_price:
            counts["price_floor"] += 1
            continue
        if adv <= float(min_adv):
            counts["adv"] += 1
            continue
        if adv_dollar <= min_dollar_volume:
            counts["dollar_volume"] += 1
            continue
        if not _passes_spy_correlation_gate(
            t,
            prices_to_asof,
            spy_ticker=spy_ticker,
            window=spy_correlation_window,
            max_corr=max_spy_correlation,
            min_obs=spy_min_observations,
        ):
            counts["spy_correlation"] += 1
            continue
        keep.append(t)
    counts["kept"] = len(keep)
    return keep, counts


def _estimate_pair_params(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
    """Estimate hedge ratio and half-life for an aligned pair series."""
    x = sm.add_constant(series_b.values)
    y = series_a.values
    beta = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    spread = series_a - beta * series_b
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    aligned = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    if len(aligned) < 20:
        return beta, np.inf
    x2 = sm.add_constant(aligned.iloc[:, 0].values)
    y2 = aligned.iloc[:, 1].values
    lam = float(np.linalg.lstsq(x2, y2, rcond=None)[0][1])
    if lam >= 0:
        return beta, np.inf
    half_life = float(-np.log(2) / lam)
    return beta, half_life


def walk_forward_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    tradable_tickers: list[str] | None = None,
    backtest_start: str,
    backtest_end: str,
    formation_days: int = 252,
    rescore_freq_trading_days: int = 21,
    n_clusters: int = CONFIG.n_clusters,
    min_coint_history: int = 200,
    entry_z: float = CONFIG.entry_z,
    exit_z: float = CONFIG.exit_z,
    stop_loss_z: float = CONFIG.stop_loss_z,
    max_holding_multiplier: float = CONFIG.max_holding_multiplier,
    initial_capital: float = CONFIG.initial_capital,
    max_active_pairs: int = CONFIG.max_active_pairs,
    target_gross_per_pair_pct: float = CONFIG.target_gross_per_pair_pct,
    max_gross_exposure_pct: float = CONFIG.max_gross_exposure_pct,
    commission_bps: float = CONFIG.commission_bps,
    slippage_bps: float = CONFIG.slippage_bps,
    short_borrow_annual: float = CONFIG.short_borrow_annual,
    trading_days_per_year: int = CONFIG.trading_days_per_year,
    vix: pd.Series | None = None,
    use_vix_filter: bool = CONFIG.use_vix_filter,
    vix_entry_block: float = CONFIG.vix_entry_block,
    vix_resume: float = CONFIG.vix_resume,
    vix_resume_days: int = CONFIG.vix_resume_days,
    use_silhouette_k_selection: bool | None = None,
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, list[dict], pd.DatetimeIndex]:
    """Run a walk-forward backtest with no in-sample leakage into OOS decisions.

    Discovery (cluster, Engle-Granger, OLS hedge, half-life) uses only prices
    with index strictly before each segment's first live day. Signals use
    lagged_z_score so decisions at t depend only on data through t-1.

    Args:
        prices: Wide adjusted close panel, DatetimeIndex rows.
        volumes: Wide daily volume panel aligned to prices index/columns.
        returns: Simple pct-change returns aligned to prices (may start one row later).
        tradable_tickers: Base ticker list before point-in-time universe gating.
        backtest_start: First date (inclusive) to include in performance stats.
        backtest_end: Last date (inclusive) for performance stats.
        formation_days: Trading days of history before each segment's first live day.
        rescore_freq_trading_days: Segment length; re-run discovery each segment.
        n_clusters: Fixed KMeans k when silhouette selection is disabled (default **5**);
            also fallback k when silhouette scan fails or ``use_silhouette_k_selection`` is False.
        min_coint_history: Minimum overlapping days required for cointegration test.
        entry_z: Entry threshold on lagged z.
        exit_z: Exit threshold on lagged z.
        stop_loss_z: Stop when |lagged z| exceeds this.
        max_holding_multiplier: Time stop when days_held exceeds this times half-life.
        vix: VIX close series aligned in time with ``prices`` index (or superset); required
            if ``use_vix_filter`` is True.
        use_vix_filter: If True, block **new** entries using lagged VIX and hysteresis.
        vix_entry_block: Lagged VIX level at or above which new entries are blocked.
        vix_resume: Lagged VIX must be at or below this for ``vix_resume_days`` to re-enable.
        vix_resume_days: Consecutive calm days required before new entries resume.
        use_silhouette_k_selection: If None, uses ``CONFIG.use_silhouette_k_selection``.
            If False, KMeans uses fixed ``n_clusters`` (default 5). If True, picks k by silhouette.
        verbose: If True, print each segment's formation vs live date ranges before clustering.

    Returns:
        Tuple of (daily portfolio returns from NAV, daily NAV series, active-pair count
        series, pair PnL table, per-pair stats, DatetimeIndex of OOS dates).
    """
    risk_manager = PairsRiskManager(
        stop_loss_z=stop_loss_z,
        max_holding_multiplier=max_holding_multiplier,
    )

    if use_vix_filter:
        if vix is None or len(vix) == 0:
            raise ValueError("use_vix_filter is True but vix series is missing or empty.")

    sil_sel = (
        CONFIG.use_silhouette_k_selection
        if use_silhouette_k_selection is None
        else use_silhouette_k_selection
    )

    cal = prices.index.sort_values()
    vix_lagged: pd.Series | None = None
    if use_vix_filter and vix is not None:
        vix_lagged = _align_vix_lagged_to_calendar(vix, cal)
    vix_entries_ok = True
    vix_streak = 0
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

    daily_borrow_rate = short_borrow_annual / float(trading_days_per_year)
    cash = float(initial_capital)
    nav = float(initial_capital)
    nav_series: dict[pd.Timestamp, float] = {}
    active_pairs_series: dict[pd.Timestamp, float] = {}
    pair_stats_map: dict[str, dict[str, int]] = {}
    pair_accounting: dict[str, dict[str, float | int]] = {}
    positions: dict[str, dict] = {}

    seg_start = first_bt_pos
    while seg_start <= last_bt_pos:
        seg_end = min(seg_start + rescore_freq_trading_days - 1, last_bt_pos)
        form_lo = seg_start - formation_days
        form_hi = seg_start  # exclusive: rows form_lo .. seg_start-1
        formation_prices = prices.iloc[form_lo:form_hi]
        formation_volumes = volumes.iloc[form_lo:form_hi]
        base_tickers = tradable_tickers or [c for c in formation_prices.columns if c != CONFIG.spy_ticker]
        eligible, gate_counts = _filter_universe_for_segment(
            formation_prices,
            formation_volumes,
            tradable_tickers=base_tickers,
            min_price=CONFIG.min_price,
            min_adv=CONFIG.min_adv,
            min_dollar_volume=CONFIG.min_dollar_volume,
            liquidity_window=CONFIG.liquidity_window,
            spy_ticker=CONFIG.spy_ticker,
            spy_correlation_window=CONFIG.spy_correlation_window,
            max_spy_correlation=CONFIG.max_spy_correlation,
            spy_min_observations=CONFIG.spy_min_observations,
        )
        if verbose:
            logger.info(
                "Universe gates @ %s | base=%d kept=%d dropped=%d | "
                "missing=%d price=%d adv=%d dollar_vol=%d spy_corr=%d",
                cal[seg_start - 1].date(),
                gate_counts["base"],
                gate_counts["kept"],
                gate_counts["base"] - gate_counts["kept"],
                gate_counts["missing_columns_or_history"],
                gate_counts["price_floor"],
                gate_counts["adv"],
                gate_counts["dollar_volume"],
                gate_counts["spy_correlation"],
            )
        formation_prices = formation_prices[eligible]
        formation_returns = calculate_returns(formation_prices)

        min_tickers = CONFIG.cluster_k_min if sil_sel else n_clusters
        if formation_returns.shape[0] < 20 or formation_returns.shape[1] < min_tickers:
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

        clusters = cluster_stocks(
            formation_returns,
            n_clusters=n_clusters,
            use_silhouette_k_selection=sil_sel,
            k_min=CONFIG.cluster_k_min,
            k_max=CONFIG.cluster_k_max,
            kmeans_n_init=CONFIG.kmeans_n_init,
            random_state=CONFIG.kmeans_random_seed,
            verbose=verbose,
        )
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
        pair_inputs: dict[str, dict] = {}
        pair_order: list[str] = []

        for _, row in tradable.iterrows():
            stock_a = row["Stock_A"]
            stock_b = row["Stock_B"]
            hedge_ratio = float(row["Hedge_Ratio"])
            half_life = float(row["Half_Life_Days"])
            pair_key = f"{stock_a}_{stock_b}"
            pair_order.append(pair_key)

            if pair_key not in pair_stats_map:
                pair_stats_map[pair_key] = {"Trades_Won": 0, "Trades_Lost": 0}
            if pair_key not in pair_accounting:
                pair_accounting[pair_key] = {
                    "cash_ledger": 0.0,
                    "fees_total": 0.0,
                    "borrow_total": 0.0,
                    "trade_count": 0,
                    "days_active": 0,
                }

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
            pair_inputs[pair_key] = {
                "stock_a": stock_a,
                "stock_b": stock_b,
                "hedge_ratio": hedge_ratio,
                "half_life": half_life,
                "z_signal": z_signal,
            }
            if pair_key not in positions:
                positions[pair_key] = {
                    "stock_a": stock_a,
                    "stock_b": stock_b,
                    "hedge_ratio": hedge_ratio,
                    "qty_a": 0.0,
                    "qty_b": 0.0,
                    "is_open": False,
                    "days_held": 0,
                    "signal": 0,
                }
            else:
                positions[pair_key]["stock_a"] = stock_a
                positions[pair_key]["stock_b"] = stock_b
                positions[pair_key]["hedge_ratio"] = hedge_ratio

        segment_keys = set(pair_order)
        for d in segment_cal:
            prices_t = prices.loc[d]

            if use_vix_filter and vix_lagged is not None:
                v_lag = float(vix_lagged.loc[d])
                vix_entries_ok, vix_streak = _step_vix_entry_permission(
                    vix_entries_ok,
                    vix_streak,
                    v_lag,
                    entry_block=vix_entry_block,
                    resume=vix_resume,
                    resume_days=vix_resume_days,
                )
                if verbose and d == segment_cal[0]:
                    v_str = f"{v_lag:.2f}" if np.isfinite(v_lag) else "nan"
                    logger.info(
                        "VIX gate @ %s | lagged_VIX=%s | entries_ok=%s streak=%s "
                        "(block>=%s resume<=%s for %s d)",
                        d.date(),
                        v_str,
                        vix_entries_ok,
                        vix_streak,
                        vix_entry_block,
                        vix_resume,
                        vix_resume_days,
                    )

            # Force-close pairs that are no longer selected this segment.
            for key, pos in positions.items():
                if key in segment_keys or not pos["is_open"]:
                    continue
                a = pos["stock_a"]
                b = pos["stock_b"]
                if a not in prices_t.index or b not in prices_t.index:
                    continue
                cash_delta, _ = _apply_target_position(
                    pos,
                    0.0,
                    0.0,
                    float(prices_t[a]),
                    float(prices_t[b]),
                    commission_bps,
                    slippage_bps,
                )
                cash += cash_delta
                pair_accounting[key]["cash_ledger"] += float(cash_delta)
                pair_accounting[key]["trade_count"] += 1
                pos["qty_a"] = 0.0
                pos["qty_b"] = 0.0
                pos["is_open"] = False
                pos["days_held"] = 0
                pos["signal"] = 0

            # Daily borrow on open short legs.
            for pos in positions.values():
                if not pos["is_open"]:
                    continue
                a = pos["stock_a"]
                b = pos["stock_b"]
                if a not in prices_t.index or b not in prices_t.index:
                    continue
                short_notional = 0.0
                if pos["qty_a"] < 0:
                    short_notional += abs(pos["qty_a"] * float(prices_t[a]))
                if pos["qty_b"] < 0:
                    short_notional += abs(pos["qty_b"] * float(prices_t[b]))
                borrow = short_notional * daily_borrow_rate
                cash -= borrow
                key = f"{a}_{b}"
                if key in pair_accounting:
                    pair_accounting[key]["cash_ledger"] -= float(borrow)
                    pair_accounting[key]["borrow_total"] += float(borrow)

            # Evaluate signals and execute at day close.
            for pair_key in pair_order:
                inp = pair_inputs[pair_key]
                pos = positions[pair_key]
                a = inp["stock_a"]
                b = inp["stock_b"]
                if a not in prices_t.index or b not in prices_t.index:
                    continue
                pa = float(prices_t[a])
                pb = float(prices_t[b])
                z = inp["z_signal"].loc[d] if d in inp["z_signal"].index else np.nan
                if pd.isna(z):
                    continue

                target_signal = pos["signal"]
                if pos["is_open"]:
                    pos["days_held"] += 1
                    if risk_manager.check_stop_loss(float(z)):
                        target_signal = 0
                        pair_stats_map[pair_key]["Trades_Lost"] += 1
                    elif risk_manager.check_time_stop(pos["days_held"], inp["half_life"]):
                        target_signal = 0
                        pair_stats_map[pair_key]["Trades_Lost"] += 1
                    elif (pos["signal"] == 1 and z >= exit_z) or (pos["signal"] == -1 and z <= exit_z):
                        target_signal = 0
                        pair_stats_map[pair_key]["Trades_Won"] += 1
                else:
                    if z < -entry_z:
                        target_signal = 1
                    elif z > entry_z:
                        target_signal = -1
                    else:
                        target_signal = 0

                # Respect portfolio capacity for new entries.
                if not pos["is_open"] and target_signal != 0:
                    open_positions = sum(1 for p in positions.values() if p["is_open"])
                    gross_now = _current_gross_exposure(positions, prices_t)
                    gross_target = nav * target_gross_per_pair_pct
                    max_gross = nav * max_gross_exposure_pct
                    if open_positions >= max_active_pairs or gross_now + gross_target > max_gross:
                        target_signal = 0

                if (
                    not pos["is_open"]
                    and target_signal != 0
                    and use_vix_filter
                    and not vix_entries_ok
                ):
                    target_signal = 0

                if target_signal == pos["signal"] and pos["is_open"] == (target_signal != 0):
                    if pos["is_open"]:
                        pair_accounting[pair_key]["days_active"] += 1
                    continue

                gross_target = nav * target_gross_per_pair_pct if target_signal != 0 else 0.0
                tqa, tqb = _target_qty_for_signal(target_signal, inp["hedge_ratio"], pa, pb, gross_target)
                cash_delta, _ = _apply_target_position(
                    pos,
                    tqa,
                    tqb,
                    pa,
                    pb,
                    commission_bps,
                    slippage_bps,
                )
                cash += cash_delta
                traded_notional = abs((tqa - pos["qty_a"]) * pa) + abs((tqb - pos["qty_b"]) * pb)
                fees = _trade_cost(traded_notional, commission_bps, slippage_bps)
                pair_accounting[pair_key]["cash_ledger"] += float(cash_delta)
                pair_accounting[pair_key]["fees_total"] += float(fees)
                pair_accounting[pair_key]["trade_count"] += 1
                pos["qty_a"] = tqa
                pos["qty_b"] = tqb
                pos["is_open"] = target_signal != 0
                pos["signal"] = target_signal
                pos["days_held"] = 1 if target_signal != 0 else 0
                if pos["is_open"]:
                    pair_accounting[pair_key]["days_active"] += 1

            nav = _mark_to_market(cash, positions, prices_t)
            nav_series[d] = nav
            active_pairs_series[d] = float(sum(1 for p in positions.values() if p["is_open"]))

        # Flatten all open positions at segment end so next segment starts clean.
        if len(segment_cal) > 0:
            d_last = segment_cal[-1]
            prices_last = prices.loc[d_last]
            for pos in positions.values():
                if not pos["is_open"]:
                    continue
                a = pos["stock_a"]
                b = pos["stock_b"]
                if a not in prices_last.index or b not in prices_last.index:
                    continue
                cash_delta, _ = _apply_target_position(
                    pos,
                    0.0,
                    0.0,
                    float(prices_last[a]),
                    float(prices_last[b]),
                    commission_bps,
                    slippage_bps,
                )
                cash += cash_delta
                key = f"{a}_{b}"
                if key in pair_accounting:
                    pair_accounting[key]["cash_ledger"] += float(cash_delta)
                    pair_accounting[key]["trade_count"] += 1
                pos["qty_a"] = 0.0
                pos["qty_b"] = 0.0
                pos["is_open"] = False
                pos["days_held"] = 0
                pos["signal"] = 0

        seg_start = seg_end + 1

    oos_index = cal[(cal >= bt_start) & (cal <= bt_end)]
    nav_oos = pd.Series(index=oos_index, dtype=float)
    running_nav = float(initial_capital)
    for d in oos_index:
        if d in nav_series:
            running_nav = nav_series[d]
        nav_oos.loc[d] = running_nav
    portfolio_returns = nav_oos.pct_change().fillna(0.0)
    active_pairs_oos = pd.Series(index=oos_index, dtype=float)
    running_active = 0.0
    for d in oos_index:
        if d in active_pairs_series:
            running_active = active_pairs_series[d]
        active_pairs_oos.loc[d] = running_active

    pair_stats = [
        {"Pair": k, "Trades_Won": v["Trades_Won"], "Trades_Lost": v["Trades_Lost"]}
        for k, v in sorted(pair_stats_map.items())
    ]
    pair_rows: list[dict] = []
    prices_last = prices.loc[oos_index[-1]] if len(oos_index) else prices.iloc[-1]
    for key, acct in sorted(pair_accounting.items()):
        pos = positions.get(key)
        mtm = 0.0
        if pos and pos["is_open"]:
            a = pos["stock_a"]
            b = pos["stock_b"]
            if a in prices_last.index and b in prices_last.index:
                mtm = pos["qty_a"] * float(prices_last[a]) + pos["qty_b"] * float(prices_last[b])
        total_pnl = float(acct["cash_ledger"]) + float(mtm)
        pair_rows.append(
            {
                "pair": key,
                "total_pnl": total_pnl,
                "realized_pnl": float(acct["cash_ledger"]),
                "unrealized_pnl": float(mtm),
                "fees_total": float(acct["fees_total"]),
                "borrow_total": float(acct["borrow_total"]),
                "trade_count": int(acct["trade_count"]),
                "days_active": int(acct["days_active"]),
            }
        )
    pair_pnl_df = pd.DataFrame(pair_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)

    return portfolio_returns, nav_oos, active_pairs_oos, pair_pnl_df, pair_stats, oos_index


def summarize_and_plot(
    portfolio_returns: pd.Series,
    nav_series: pd.Series,
    active_pairs_series: pd.Series,
    pair_pnl_df: pd.DataFrame,
    pair_stats: list[dict],
    *,
    backtest_start: str,
    backtest_end: str,
    plot_path: str = "portfolio_performance.png",
    training_start: str | None = None,
    timeline_plot_path: str = "portfolio_timeline.png",
) -> None:
    """Print summary metrics and save cumulative return plot.

    Args:
        portfolio_returns: Daily OOS returns.
        nav_series: Daily OOS NAV in dollars.
        active_pairs_series: Daily count of active pairs.
        pair_pnl_df: Pair-level PnL table sorted by total_pnl descending.
        pair_stats: Per-pair win/loss counts from walk_forward_backtest.
        backtest_start: Label for printout.
        backtest_end: Label for printout.
        plot_path: Output path for PNG.
        training_start: Optional panel/training start date for timeline plot.
        timeline_plot_path: Output path for training+OOS timeline chart.
    """
    cumulative_returns = nav_series / nav_series.iloc[0]
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) else 0.0

    excess_returns = portfolio_returns
    if excess_returns.std() > 0:
        sharpe_ratio = float(np.sqrt(252) * (excess_returns.mean() / excess_returns.std()))
    else:
        sharpe_ratio = 0.0

    rolling_max = nav_series.cummax()
    drawdown = nav_series / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    total_won = sum(p["Trades_Won"] for p in pair_stats)
    total_lost = sum(p["Trades_Lost"] for p in pair_stats)
    total_trades = total_won + total_lost
    win_rate = total_won / total_trades if total_trades > 0 else 0.0

    print("\n=========================================")
    print("   WALK-FORWARD PORTFOLIO (NO LOOK-AHEAD)")
    print("=========================================")
    print(f"OOS window:         {backtest_start} .. {backtest_end}")
    print(f"Starting NAV:       ${nav_series.iloc[0]:,.2f}")
    print(f"Ending NAV:         ${nav_series.iloc[-1]:,.2f}")
    print(f"Total Return:       {total_return * 100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_drawdown * 100:.2f}%")
    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate * 100:.2f}%")
    print(f"Avg Active Pairs:   {active_pairs_series.mean():.2f}")
    print(f"Max Active Pairs:   {int(active_pairs_series.max())}")
    print("=========================================")
    if not pair_pnl_df.empty:
        print("\nTop 5 pair contributors:")
        print(pair_pnl_df[["pair", "total_pnl", "trade_count"]].head(5).to_string(index=False))
        print("\nBottom 5 pair contributors:")
        print(pair_pnl_df[["pair", "total_pnl", "trade_count"]].tail(5).to_string(index=False))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    nav_series.plot(ax=axes[0], color="teal", label="NAV ($)", linewidth=2)
    axes[0].set_title("Walk-forward portfolio NAV (out-of-sample)")
    axes[0].set_ylabel("NAV ($)")
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].legend()
    active_pairs_series.plot(ax=axes[1], color="darkorange", label="Active Pairs", linewidth=1.75)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Active Pair Count Over Time")
    axes[1].grid(True, linestyle="--", alpha=0.7)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Portfolio performance graph saved to {plot_path!r}")

    if not pair_pnl_df.empty:
        top = pair_pnl_df.head(10).sort_values("total_pnl", ascending=True)
        bottom = pair_pnl_df.tail(10).sort_values("total_pnl", ascending=True)
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        axes2[0].barh(top["pair"], top["total_pnl"], color="seagreen")
        axes2[0].set_title("Top 10 Pair Contributors")
        axes2[0].set_xlabel("PnL ($)")
        axes2[1].barh(bottom["pair"], bottom["total_pnl"], color="firebrick")
        axes2[1].set_title("Bottom 10 Pair Contributors")
        axes2[1].set_xlabel("PnL ($)")
        plt.tight_layout()
        pnl_plot_path = plot_path.replace(".png", "_pair_pnl.png")
        plt.savefig(pnl_plot_path)
        plt.close()
        print(f"Pair PnL chart saved to {pnl_plot_path!r}")

    if training_start is not None:
        t0 = pd.Timestamp(training_start)
        bt0 = pd.Timestamp(backtest_start)
        bt1 = pd.Timestamp(backtest_end)
        full_index = pd.date_range(start=t0, end=bt1, freq="B")
        timeline_nav = pd.Series(nav_series.iloc[0], index=full_index, dtype=float)
        timeline_nav.loc[nav_series.index] = nav_series.values

        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
        timeline_nav.plot(ax=ax3, color="teal", linewidth=2, label="NAV timeline")
        ax3.axvline(bt0, color="black", linestyle="--", linewidth=1.2, label="Backtest start")
        ax3.axvspan(t0, bt0, color="lightgray", alpha=0.35, label="Training / formation only")
        ax3.set_title("Portfolio timeline: training period and OOS period")
        ax3.set_ylabel("NAV ($)")
        ax3.set_xlabel("Date")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend()
        plt.tight_layout()
        plt.savefig(timeline_plot_path)
        plt.close()
        print(f"Training-to-OOS timeline graph saved to {timeline_plot_path!r}")


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
