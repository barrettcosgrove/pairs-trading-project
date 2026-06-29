# Central simulation loop iterating day by day over the full configured range.
# Monthly: universe reconstitution. Quarterly: re-cluster and composite scoring.
# Daily: regime filters, signals, execution, costs. Writes trade_log and nav_series.
# Entry point for scripts/03_run_backtest.py; scripts/04 slices OOS from outputs.

import logging
import math
from datetime import date

import numpy as np
import pandas as pd

from src.backtest.costs import accrue_borrow_cost, apply_costs
from src.backtest.execution import execute
from src.backtest.portfolio import Portfolio, Position
from src.clustering.correlation import build_distance_matrix
from src.clustering.kmeans import run_clustering
from src.config import StrategyConfig
from src.data.load import load_prices, load_returns, load_universe, load_vix
from src.regime.earnings import in_blackout
from src.regime.vix import new_entries_permitted
from src.scoring.composite import score_candidates
from src.signals.entry_exit import get_signal

logger = logging.getLogger(__name__)


def run_backtest(
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the ARQ pairs trading simulation over the full backtest date range.

    Iterates every trading day from ``backtest_start_date`` through
    ``backtest_end_date`` with a single continuous portfolio (no reset at the
    OOS calendar boundary). ``oos_fraction`` is reserved for reporting in
    ``scripts/04_walkforward.py`` only.

    Args:
        config: StrategyConfig instance. Use CONFIG for the default run
                or pass a modified instance for sensitivity analysis.

    Returns:
        Tuple ``(trade_log, nav_series, pair_daily_mtm)`` where:

        trade_log columns:
            date, ticker_a, ticker_b, action, shares_a, shares_b,
            price_a, price_b, cost, pnl,
            dollar_allocation_at_entry, realized_net_usd, return_pct
            (pnl = spread price P&L; on exits realized_net_usd = pnl minus exit
            cost minus entry fee; return_pct = realized_net_usd divided by
            long-leg dollar allocation times 100. Opens: realized_net_usd 0,
            return_pct NaN.)

        nav_series columns:
            date, nav, cash, gross_exposure, drawdown_from_peak

        pair_daily_mtm columns (EOD after trades, borrow; one row per open pair):
            date, ticker_a, ticker_b, direction, cluster_id, mtm_usd,
            gross_exposure_pair, dollar_allocation_at_entry, portfolio_nav_post_trade
    """
    # ── Load full dataset ─────────────────────────────────────────────────────
    start_date = pd.Timestamp(config.backtest_start_date).date() if getattr(config, 'backtest_start_date', None) else None
    end_date = pd.Timestamp(config.backtest_end_date).date() if getattr(config, 'backtest_end_date', None) else None
    all_prices  = load_prices(start=start_date, end=end_date)
    all_returns = load_returns(start=start_date, end=end_date)
    vix_series  = load_vix(start=start_date, end=end_date)

    all_dates = sorted(all_prices["date"].unique())
    run_dates = all_dates

    if not run_dates:
        logger.warning(
            "No trading days in loaded price range (check backtest_start_date / "
            "backtest_end_date and data availability) — returning empty outputs."
        )
        return _empty_backtest_outputs()

    logger.info(
        "Backtest period: %s → %s (%d trading days; OOS slice uses oos_fraction=%.2f in script 04)",
        run_dates[0].date(),
        run_dates[-1].date(),
        len(run_dates),
        config.oos_fraction,
    )

    # ── State ─────────────────────────────────────────────────────────────────
    portfolio         = Portfolio(config=config)
    trade_log_rows    = []
    nav_rows          = []
    pair_daily_rows   = []
    active_pairs      = {}          # { (ticker_a, ticker_b): {formation_stats} }
    cluster_map       = {}          # (ticker_a, ticker_b) -> cluster_id
    pending_retries   = []          # orders to retry next day
    stoploss_cooldown: dict[tuple[str, str], int] = {}
    candidates        = pd.DataFrame()

    last_universe_date = None
    last_cluster_date = None
    prior_week_date = None

    # ── Daily loop ────────────────────────────────────────────────────────────
    for i, today in enumerate(run_dates):
        today_date = pd.Timestamp(today).date()
        logger.debug("── Day %d: %s ──", i, today_date)

        prices_to_date  = all_prices[all_prices["date"] <= today]
        returns_to_date = all_returns[all_returns["date"] <= today]

        # ── Weekly NAV snapshot for soft drawdown check ───────────────────────
        if prior_week_date is None or (today_date - prior_week_date).days >= 7:
            if i > 0:
                portfolio.update_prior_week_nav(float(nav_rows[-1]["nav"]))
            prior_week_date = today_date

        # Decrement pair cooldown counters once per trading day.
        for pair in list(stoploss_cooldown.keys()):
            stoploss_cooldown[pair] -= 1
            if stoploss_cooldown[pair] <= 0:
                del stoploss_cooldown[pair]

        # ── Monthly: universe reconstitution ──────────────────────────────────
        if (last_universe_date is None or
                (today_date - last_universe_date).days >= config.universe_refresh_days):
            universe = load_universe(as_of=today_date)
            last_universe_date = today_date
            logger.info(
                "Universe reconstituted on %s — %d tickers", today_date, len(universe)
            )

        # ── Quarterly: re-cluster and re-score ────────────────────────────────
        if (last_cluster_date is None or
                (today_date - last_cluster_date).days >= config.clustering_refresh_days):
            try:
                distance_matrix = build_distance_matrix(
                    returns_to_date[returns_to_date["ticker"].isin(universe)],
                    window=config.clustering_window,
                )
                clusters   = run_clustering(distance_matrix)
                candidates = score_candidates(
                    clusters,
                    returns_to_date[returns_to_date["ticker"].isin(universe)],
                    prices_to_date[prices_to_date["ticker"].isin(universe)],
                    today_date,
                )
                
                # Immediately extract the active pairs from the finalists
                if not candidates.empty:
                    active_pairs = {}
                    cluster_map.clear()
                    for _, row in candidates.iterrows():
                        pair = (row["ticker_a"], row["ticker_b"])
                        active_pairs[pair] = {
                            "beta_formation": float(row["beta_formation"]),
                            "mean_formation": float(row["mean_formation"]),
                            "std_formation":  float(row["std_formation"]),
                            "expected_halflife": float(row["halflife_value"])
                        }
                        cluster_map[pair] = row["cluster_id"]
                else:
                    active_pairs = {}
                    cluster_map.clear()
                    
                last_cluster_date = today_date
                logger.info(
                    "Clustering complete on %s — %d clusters, %d candidates",
                    today_date, len(clusters), len(candidates),
                )
            except Exception as exc:
                logger.error("Clustering failed on %s: %s — skipping", today_date, exc)
                candidates = pd.DataFrame()

        # ── NAV and drawdown ──────────────────────────────────────────────────
        nav = portfolio.compute_nav(prices_to_date, today_date)
        dd  = portfolio.drawdown_from_peak(nav)

        price_today_map = (
            prices_to_date[prices_to_date["date"] == today]
            .set_index("ticker")["adj_close"]
            .to_dict()
        )

        gross_exposure = sum(
            abs(pos.shares_a) * price_today_map.get(pos.ticker_a, 0.0)
            + abs(pos.shares_b) * price_today_map.get(pos.ticker_b, 0.0)
            for pos in portfolio.positions.values()
        )

        nav_rows.append({
            "date":               today_date,
            "nav":                nav,
            "cash":               portfolio.cash,
            "gross_exposure":     gross_exposure,
            "drawdown_from_peak": dd,
        })

        entries_ok, size_factor = portfolio.check_drawdown_controls(nav)

        # ── Gradual trim on hard halt ─────────────────────────────────────────
        if portfolio.needs_trim():
            _execute_trim(portfolio, prices_to_date, today_date, config, trade_log_rows)
            portfolio.decrement_trim_day()

        # ── Portfolio-level regime filter ─────────────────────────────────────
        vix_ok = new_entries_permitted(today_date, vix_series)

        # ── Retry queue from previous day ─────────────────────────────────────
        still_pending = []
        for retry_order in pending_retries:
            fill = execute(retry_order, prices_to_date, today_date)
            if fill["success"]:
                _record_open(
                    fill,
                    retry_order,
                    portfolio,
                    cluster_map,
                    trade_log_rows,
                    today_date,
                    prices_to_date,
                )
            elif fill["retry_next_day"]:
                still_pending.append(retry_order)
        pending_retries = still_pending

        # ── Process all active pairs + any open positions ──────────────────────
        open_pairs = list(portfolio.positions.keys())
        pairs_to_process = list(set(active_pairs.keys()).union(open_pairs))

        for ticker_a, ticker_b in pairs_to_process:
            pair     = (ticker_a, ticker_b)
            open_pos = portfolio.positions.get(pair)
            days_open        = open_pos.days_open  if open_pos else 0
            current_position = open_pos.direction  if open_pos else None
            
            if open_pos:
                beta_formation = open_pos.beta_at_entry
                mean_formation = open_pos.mean_at_entry
                std_formation  = open_pos.std_at_entry
                expected_halflife = open_pos.expected_halflife
                
                # ── Dynamic Beta Rebalancing ──
                # If current beta drifts from formation beta, adjust shares_b
                current_beta = _compute_live_beta(
                    ticker_a, ticker_b, prices_to_date, config
                )
                if current_beta is not None and abs(current_beta - open_pos.beta_at_entry) > config.beta_rebalance_threshold:
                    logger.info("Beta rebalance %s/%s: %.3f -> %.3f", ticker_a, ticker_b, open_pos.beta_at_entry, current_beta)
                    _rebalance_beta(
                        portfolio,
                        open_pos,
                        current_beta,
                        price_today_map.get(ticker_b, 0.0),
                    )
                    
            elif pair in active_pairs:
                beta_formation = active_pairs[pair]["beta_formation"]
                mean_formation = active_pairs[pair]["mean_formation"]
                std_formation  = active_pairs[pair]["std_formation"]
                expected_halflife = active_pairs[pair]["expected_halflife"]
            else:
                continue # Should not happen

            # ── Signal ───────────────────────────────────────────────────────
            signal = get_signal(
                ticker_a,
                ticker_b,
                prices_to_date,
                today_date,
                beta_formation=beta_formation,
                mean_formation=mean_formation,
                std_formation=std_formation,
                expected_halflife=expected_halflife,
                days_open=days_open,
                current_position=current_position,
                config=config,
            )

            price_a = price_today_map.get(ticker_a)
            price_b = price_today_map.get(ticker_b)

            if price_a is None or price_b is None:
                logger.warning(
                    "Missing price for %s or %s on %s", ticker_a, ticker_b, today_date
                )
                continue

            # ── Exit signals ──────────────────────────────────────────────────
            if signal in ("TAKE_PROFIT", "STOP_LOSS", "TIME_STOP") and open_pos:
                _close_pair(
                    portfolio, ticker_a, ticker_b, price_a, price_b,
                    signal, trade_log_rows, today_date, prices_to_date,
                )
                if signal == "STOP_LOSS":
                    stoploss_cooldown[(ticker_a, ticker_b)] = config.pair_stop_cooldown_days
                continue

            # ── Entry signals ─────────────────────────────────────────────────
            if signal in ("LONG_SPREAD", "SHORT_SPREAD") and open_pos is None:
                if not entries_ok:
                    logger.info("Entry blocked by drawdown halt: %s/%s", ticker_a, ticker_b)
                    continue
                if not vix_ok:
                    logger.info("Entry blocked by VIX filter: %s/%s", ticker_a, ticker_b)
                    continue
                if in_blackout(ticker_a, today_date) or in_blackout(ticker_b, today_date):
                    logger.info(
                        "Entry blocked by earnings blackout: %s/%s", ticker_a, ticker_b
                    )
                    continue

                # Only open if it's currently an active pair
                if (ticker_a, ticker_b) not in active_pairs:
                    continue
                if stoploss_cooldown.get((ticker_a, ticker_b), 0) > 0:
                    logger.info(
                        "Entry blocked by stop-loss cooldown: %s/%s (%d day(s) left)",
                        ticker_a,
                        ticker_b,
                        stoploss_cooldown[(ticker_a, ticker_b)],
                    )
                    continue

                if not portfolio.can_open(ticker_a, ticker_b, len(active_pairs)):
                    continue
                    
                beta = beta_formation
                mean = mean_formation
                std  = std_formation

                dollar_alloc = portfolio.position_size(nav, size_factor, len(active_pairs))
                shares_a_raw = dollar_alloc / price_a
                shares_b_raw = (dollar_alloc * beta) / price_b

                if signal == "LONG_SPREAD":
                    shares_a, shares_b = shares_a_raw, -shares_b_raw
                else:
                    shares_a, shares_b = -shares_a_raw, shares_b_raw

                order = {
                    "ticker_a": ticker_a, "ticker_b": ticker_b,
                    "shares_a": shares_a, "shares_b": shares_b,
                    "action":   "OPEN",
                    "signal":   signal,
                    "beta":     beta,
                    "mean":     mean,
                    "std":      std,
                    "expected_halflife": expected_halflife,
                    "dollar_alloc": dollar_alloc
                }
                fill = execute(order, prices_to_date, today_date)

                if fill["retry_next_day"]:
                    pending_retries.append(order)
                    continue

                if fill["success"]:
                    trade_dict = {
                        "ticker_a":        ticker_a,
                        "ticker_b":        ticker_b,
                        "shares_a":        fill["filled_a"],
                        "shares_b":        fill["filled_b"],
                        "beta":            beta,
                        "mean":            mean,
                        "std":             std,
                        "expected_halflife": expected_halflife,
                        "price_a":         fill["price_a"],
                        "price_b":         fill["price_b"],
                        "adv_a":           _get_adv(ticker_a, prices_to_date),
                        "adv_b":           _get_adv(ticker_b, prices_to_date),
                        "expected_profit": dollar_alloc * std,
                    }
                    cost = apply_costs(trade_dict)
                    if cost == -1.0:
                        logger.info(
                            "Trade skipped: profit-to-cost gate %s/%s", ticker_a, ticker_b
                        )
                        continue

                    portfolio.cash -= cost
                    cluster_id = cluster_map.get(pair, -1)
                    portfolio.open_position(
                        ticker_a,
                        ticker_b,
                        cluster_id,
                        signal,
                        beta,
                        mean,
                        std,
                        expected_halflife,
                        fill["price_a"],
                        fill["price_b"],
                        dollar_alloc,
                        today_date,
                        entry_transaction_cost=cost,
                    )
                    trade_log_rows.append({
                        "date":     today_date,
                        "ticker_a": ticker_a, "ticker_b": ticker_b,
                        "action":   signal,
                        "shares_a": fill["filled_a"], "shares_b": fill["filled_b"],
                        "price_a":  fill["price_a"],  "price_b":  fill["price_b"],
                        "cost":     cost, "pnl": 0.0,
                        "dollar_allocation_at_entry": float(dollar_alloc),
                        "realized_net_usd": 0.0,
                        "return_pct": math.nan,
                    })

        # ── Daily borrow accrual ──────────────────────────────────────────────
        for pos in portfolio.positions.values():
            short_ticker = pos.ticker_a if pos.shares_a < 0 else pos.ticker_b
            short_shares = abs(pos.shares_a if pos.shares_a < 0 else pos.shares_b)
            short_price  = price_today_map.get(short_ticker, 0.0)
            portfolio.cash -= accrue_borrow_cost(short_shares, short_price)

        portfolio.increment_days_open()

        _append_eod_pair_snapshots(
            portfolio, price_today_map, today_date, pair_daily_rows
        )

    # ── Assemble outputs ──────────────────────────────────────────────────────
    trade_log  = pd.DataFrame(trade_log_rows)
    nav_series = pd.DataFrame(nav_rows)
    pair_daily = (
        pd.DataFrame(pair_daily_rows)
        if pair_daily_rows
        else pd.DataFrame(
            columns=[
                "date",
                "ticker_a",
                "ticker_b",
                "direction",
                "cluster_id",
                "mtm_usd",
                "gross_exposure_pair",
                "dollar_allocation_at_entry",
                "portfolio_nav_post_trade",
            ]
        )
    )

    logger.info(
        "Backtest complete — %d trades, final NAV $%.0f",
        len(trade_log),
        nav_series["nav"].iloc[-1] if not nav_series.empty else 0,
    )

    return trade_log, nav_series, pair_daily


# ── Helper functions ──────────────────────────────────────────────────────────


def _empty_backtest_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build empty trade log, NAV series, and pair-daily frames with correct columns.

    Returns:
        Tuple of three empty DataFrames matching ``run_backtest`` outputs.
    """
    trade_log = pd.DataFrame(
        columns=[
            "date",
            "ticker_a",
            "ticker_b",
            "action",
            "shares_a",
            "shares_b",
            "price_a",
            "price_b",
            "cost",
            "pnl",
            "dollar_allocation_at_entry",
            "realized_net_usd",
            "return_pct",
        ]
    )
    nav_series = pd.DataFrame(
        columns=["date", "nav", "cash", "gross_exposure", "drawdown_from_peak"]
    )
    pair_daily = pd.DataFrame(
        columns=[
            "date",
            "ticker_a",
            "ticker_b",
            "direction",
            "cluster_id",
            "mtm_usd",
            "gross_exposure_pair",
            "dollar_allocation_at_entry",
            "portfolio_nav_post_trade",
        ]
    )
    return trade_log, nav_series, pair_daily


def _append_eod_pair_snapshots(
    portfolio: Portfolio,
    price_today_map: dict[str, float],
    today_date: date,
    pair_daily_rows: list,
) -> None:
    """
    Record end-of-day post-trade MTM for each open pair (after borrow accrual).

    ``mtm_usd`` is shares·price for both legs (near zero at entry; tracks
    unrealized drift). ``portfolio_nav_post_trade`` is the same for all rows
    on a given date (cash + sum of pair MTMs).

    Args:
        portfolio: Portfolio after the day's trading and borrow accrual.
        price_today_map: Today's close by ticker.
        today_date: Calendar date for the snapshot.
        pair_daily_rows: Mutable list of row dicts appended in-place.

    Returns:
        None.
    """
    if not portfolio.positions:
        return

    day_rows: list[dict] = []
    mtm_total = 0.0
    for pos in portfolio.positions.values():
        pa = float(price_today_map.get(pos.ticker_a, 0.0))
        pb = float(price_today_map.get(pos.ticker_b, 0.0))
        mtm_usd = pos.shares_a * pa + pos.shares_b * pb
        mtm_total += mtm_usd
        gross = abs(pos.shares_a) * pa + abs(pos.shares_b) * pb
        day_rows.append(
            {
                "date": today_date,
                "ticker_a": pos.ticker_a,
                "ticker_b": pos.ticker_b,
                "direction": pos.direction,
                "cluster_id": pos.cluster_id,
                "mtm_usd": mtm_usd,
                "gross_exposure_pair": gross,
                "dollar_allocation_at_entry": pos.dollar_allocation_at_entry,
            }
        )

    nav_post = portfolio.cash + mtm_total
    for row in day_rows:
        row["portfolio_nav_post_trade"] = nav_post
        pair_daily_rows.append(row)


def _close_pair(
    portfolio: Portfolio,
    ticker_a: str,
    ticker_b: str,
    price_a: float,
    price_b: float,
    reason: str,
    trade_log_rows: list,
    today_date: date,
    prices_to_date: pd.DataFrame,
) -> None:
    """
    Close an open position, apply costs, and record the trade.

    Args:
        portfolio: Live Portfolio instance.
        ticker_a: Leg A ticker.
        ticker_b: Leg B ticker.
        price_a: Exit price for leg A.
        price_b: Exit price for leg B.
        reason: Exit signal string (e.g. "TAKE_PROFIT").
        trade_log_rows: Mutable list to append the trade record to.
        today_date: Execution date.
        prices_to_date: Prices DataFrame up to today for ADV lookup.
    """
    pos = portfolio.positions.get((ticker_a, ticker_b))
    if pos is None:
        return

    order = {
        "ticker_a": ticker_a, "ticker_b": ticker_b,
        "shares_a": -pos.shares_a,
        "shares_b": -pos.shares_b,
        "action":   "CLOSE",
    }
    fill = execute(order, prices_to_date, today_date)
    if not fill["success"]:
        logger.warning(
            "Close execution failed for %s/%s on %s", ticker_a, ticker_b, today_date
        )
        return

    trade_dict = {
        "ticker_a":        ticker_a, "ticker_b": ticker_b,
        "shares_a":        abs(fill["filled_a"]),
        "shares_b":        abs(fill["filled_b"]),
        "beta":            pos.beta_at_entry,
        "mean":            pos.mean_at_entry,
        "std":             pos.std_at_entry,
        "expected_halflife": pos.expected_halflife,
        "price_a":         fill["price_a"],
        "price_b":         fill["price_b"],
        "adv_a":           _get_adv(ticker_a, prices_to_date),
        "adv_b":           _get_adv(ticker_b, prices_to_date),
        "expected_profit": float("inf"),  # exits always pass cost gate
    }
    cost = apply_costs(trade_dict)
    exit_cost = float(cost) if cost > 0 else 0.0
    if exit_cost > 0:
        portfolio.cash -= exit_cost

    alloc = float(pos.dollar_allocation_at_entry)
    entry_fee = float(pos.entry_transaction_cost)

    pnl = portfolio.close_position(ticker_a, ticker_b, price_a, price_b, reason)

    realized_net = float(pnl) - exit_cost - entry_fee
    ret_pct = (
        (realized_net / alloc * 100.0) if alloc > 1e-12 else math.nan
    )

    trade_log_rows.append({
        "date":     today_date,
        "ticker_a": ticker_a, "ticker_b": ticker_b,
        "action":   reason,
        "shares_a": fill["filled_a"], "shares_b": fill["filled_b"],
        "price_a":  fill["price_a"],  "price_b":  fill["price_b"],
        "cost":     exit_cost,
        "pnl":      pnl,
        "dollar_allocation_at_entry": alloc,
        "realized_net_usd": realized_net,
        "return_pct": ret_pct,
    })


def _execute_trim(
    portfolio: Portfolio,
    prices_to_date: pd.DataFrame,
    today_date: date,
    config: StrategyConfig,
    trade_log_rows: list,
) -> None:
    """
    Trim all positions to CONFIG.drawdown_trim_factor on hard halt.

    Args:
        portfolio: Live Portfolio instance.
        prices_to_date: Prices DataFrame up to today.
        today_date: Execution date.
        config: StrategyConfig instance.
        trade_log_rows: Mutable list — trim trades are not logged separately.
    """
    price_map = (
        prices_to_date[prices_to_date["date"] == pd.Timestamp(today_date)]
        .set_index("ticker")["adj_close"]
        .to_dict()
    )
    for pos in list(portfolio.positions.values()):
        target_shares_a = pos.shares_a * config.drawdown_trim_factor
        target_shares_b = pos.shares_b * config.drawdown_trim_factor
        trim_a = pos.shares_a - target_shares_a
        trim_b = pos.shares_b - target_shares_b

        price_a = price_map.get(pos.ticker_a, 0.0)
        price_b = price_map.get(pos.ticker_b, 0.0)

        # Signed net cash from unwinding trim_a / trim_b at today's close.
        # abs() incorrectly forced every trim to inject cash upward and inflated NAV.
        # Old (buggy) line — restore by replacing the two lines below with:
        # portfolio.cash += abs(trim_a * price_a + trim_b * price_b)
        cash_delta_trim = trim_a * price_a + trim_b * price_b
        portfolio.cash += cash_delta_trim
        pos.cumulative_cash_adjustments += cash_delta_trim
        pos.shares_a = target_shares_a
        pos.shares_b = target_shares_b

        logger.info(
            "Trim %s/%s — reduced to %.1f%% of position size",
            pos.ticker_a, pos.ticker_b, config.drawdown_trim_factor * 100,
        )


def _get_adv(ticker: str, prices_to_date: pd.DataFrame) -> float:
    """
    Compute 30-day average daily volume for a ticker.

    Args:
        ticker: Ticker symbol.
        prices_to_date: Prices DataFrame up to today.

    Returns:
        30-day ADV in shares, or 0.0 if data is unavailable.
    """
    ticker_prices = prices_to_date[prices_to_date["ticker"] == ticker]
    if ticker_prices.empty:
        return 0.0
    return ticker_prices["volume"].tail(30).mean()

def _compute_live_beta(
    ticker_a: str,
    ticker_b: str,
    prices_to_date: pd.DataFrame,
    config: StrategyConfig,
) -> float | None:
    """
    Estimate trailing OLS hedge ratio log(A) ~ log(B) over ``signal_window`` days.

    Uses only prices on or before the last row in ``prices_to_date`` (caller
    passes data truncated to the simulation day).

    Args:
        ticker_a: Dependent leg ticker.
        ticker_b: Independent leg ticker.
        prices_to_date: Long-form prices through the valuation day.
        config: Strategy parameters (uses ``signal_window``).

    Returns:
        OLS slope on log prices, or None if history is shorter than the window.
    """
    window = config.signal_window
    sub_a = prices_to_date[prices_to_date["ticker"] == ticker_a].sort_values("date")
    sub_b = prices_to_date[prices_to_date["ticker"] == ticker_b].sort_values("date")
    prices_a = sub_a.tail(window)["adj_close"].to_numpy(dtype=float)
    prices_b = sub_b.tail(window)["adj_close"].to_numpy(dtype=float)

    if len(prices_a) < window or len(prices_b) < window:
        return None

    log_a = np.log(prices_a)
    log_b = np.log(prices_b)

    x_mat = np.column_stack([np.ones(len(log_b)), log_b])
    coeffs, _, _, _ = np.linalg.lstsq(x_mat, log_a, rcond=None)
    return float(coeffs[1])


def _rebalance_beta(
    portfolio: Portfolio,
    pos: Position,
    new_beta: float,
    price_b: float,
) -> None:
    """
    Resize leg B so dollar exposure matches an updated hedge ratio.

    Args:
        portfolio: Active portfolio (mutates cash and position).
        pos: Open position to adjust.
        new_beta: Updated OLS hedge ratio.
        price_b: Same-day close for ticker B.

    Returns:
        None.
    """
    if price_b <= 0:
        return
        
    # We want dollar neutral. The long leg dictates the base value.
    # pos.shares_a * price_a + pos.shares_b * price_b = 0
    # Wait, the rule is shares_b = -(shares_a * price_a * new_beta) / price_b
    # Alternatively, just scale shares_b by the beta ratio.
    old_shares_b = pos.shares_b
    target_shares_b = pos.shares_b * (new_beta / pos.beta_at_entry) if pos.beta_at_entry != 0 else 0
    
    if target_shares_b == 0:
        return
        
    # Calculate difference
    diff_shares = target_shares_b - old_shares_b
    cash_impact = -diff_shares * price_b  # Buying shares costs cash, selling adds cash
    
    portfolio.cash += cash_impact
    pos.cumulative_cash_adjustments += cash_impact
    pos.shares_b = target_shares_b
    pos.beta_at_entry = new_beta # Update anchor to prevent constant rebalancing


def _record_open(
    fill: dict,
    order: dict,
    portfolio: Portfolio,
    cluster_map: dict,
    trade_log_rows: list,
    today_date: date,
    prices_to_date: pd.DataFrame,
) -> None:
    """
    Finalize a retry open with costs and correct ``open_position`` arguments.

    Args:
        fill: Fill result dict from ``execution.execute()`` with success True.
        order: Original order dict (signal, beta, mean, std, dollar_alloc,
            expected_halflife).
        portfolio: Live portfolio instance.
        cluster_map: Map ``(ticker_a, ticker_b)`` to cluster id.
        trade_log_rows: Mutable list of trade records.
        today_date: Execution date.
        prices_to_date: Prices through ``today_date`` for ADV lookup.

    Returns:
        None. Skips opening if the profit-to-cost gate fails or half-life is
        missing from ``order``.
    """
    ticker_a = fill["ticker_a"]
    ticker_b = fill["ticker_b"]
    cluster_id = cluster_map.get((ticker_a, ticker_b), -1)

    signal = order.get("signal", "OPEN")
    beta = float(order.get("beta", 1.0))
    mean = float(order.get("mean", 0.0))
    std = float(order.get("std", 1.0))
    dollar_alloc = float(order.get("dollar_alloc", 0.0))
    expected_halflife = order.get("expected_halflife")

    if expected_halflife is None or expected_halflife != expected_halflife:
        logger.error(
            "Retry open aborted — missing expected_halflife for %s/%s on %s",
            ticker_a,
            ticker_b,
            today_date,
        )
        return

    trade_dict = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "shares_a": fill["filled_a"],
        "shares_b": fill["filled_b"],
        "beta": beta,
        "mean": mean,
        "std": std,
        "expected_halflife": float(expected_halflife),
        "price_a": fill["price_a"],
        "price_b": fill["price_b"],
        "adv_a": _get_adv(ticker_a, prices_to_date),
        "adv_b": _get_adv(ticker_b, prices_to_date),
        "expected_profit": dollar_alloc * std,
    }
    cost = apply_costs(trade_dict)
    if cost == -1.0:
        logger.info(
            "Retry open skipped: profit-to-cost gate %s/%s on %s",
            ticker_a,
            ticker_b,
            today_date,
        )
        return

    portfolio.cash -= cost
    portfolio.open_position(
        ticker_a,
        ticker_b,
        cluster_id,
        signal,
        beta,
        mean,
        std,
        float(expected_halflife),
        fill["price_a"],
        fill["price_b"],
        dollar_alloc,
        today_date,
        entry_transaction_cost=cost,
    )

    trade_log_rows.append(
        {
            "date": today_date,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "action": signal,
            "shares_a": fill["filled_a"],
            "shares_b": fill["filled_b"],
            "price_a": fill["price_a"],
            "price_b": fill["price_b"],
            "cost": cost,
            "pnl": 0.0,
            "dollar_allocation_at_entry": float(dollar_alloc),
            "realized_net_usd": 0.0,
            "return_pct": math.nan,
        }
    )