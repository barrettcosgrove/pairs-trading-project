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
from src.signals.entry_exit import _is_momentum_breakout, get_signal
from src.signals.spread import compute as compute_spread

logger = logging.getLogger(__name__)


def run_backtest(
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        Tuple ``(trade_log, nav_series, pair_daily_mtm, blocked_entries)`` where:

        trade_log columns:
            date, ticker_a, ticker_b, action, shares_a, shares_b,
            price_a, price_b, cost, pnl,
            dollar_allocation_at_entry, realized_net_usd, return_pct
            (pnl = spread price P&L; on exits realized_net_usd = pnl minus exit
            cost minus entry fee; return_pct = realized_net_usd divided by
            long-leg dollar allocation times 100. Opens: realized_net_usd 0,
            return_pct NaN. Exit actions: TAKE_PROFIT, STOP_LOSS, TIME_STOP,
            and DOLLAR_STOP when max_pair_loss_pct is breached.)

        nav_series columns:
            date, nav, cash, gross_exposure, drawdown_from_peak

        pair_daily_mtm columns (EOD after trades, borrow; one row per open pair):
            date, ticker_a, ticker_b, direction, cluster_id, mtm_usd,
            gross_exposure_pair, dollar_allocation_at_entry, portfolio_nav_post_trade

        blocked_entries columns (one row per pair-day an actionable entry was
        suppressed): date, ticker_a, ticker_b, signal, reason — reason is one
        of drawdown_halt, vix, earnings_blackout, not_active, cooldown,
        capacity, beta, cost_gate, momentum, no_cross.
    """
    # ── Load full dataset ─────────────────────────────────────────────────────
    # Load all available history so clustering / Johansen have warmup. The
    # simulation loop itself is restricted to [backtest_start_date, backtest_end_date].
    sim_start = (
        pd.Timestamp(config.backtest_start_date)
        if getattr(config, "backtest_start_date", None)
        else None
    )
    sim_end = (
        pd.Timestamp(config.backtest_end_date)
        if getattr(config, "backtest_end_date", None)
        else None
    )
    all_prices  = load_prices(start=None, end=sim_end.date() if sim_end is not None else None)
    all_returns = load_returns(start=None, end=sim_end.date() if sim_end is not None else None)
    vix_series  = load_vix(start=None, end=sim_end.date() if sim_end is not None else None)

    all_dates = sorted(all_prices["date"].unique())
    run_dates = [
        d for d in all_dates
        if (sim_start is None or pd.Timestamp(d) >= sim_start)
        and (sim_end is None or pd.Timestamp(d) <= sim_end)
    ]

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
    blocked_rows      = []          # entry-block attribution (date, pair, reason)
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
                
                # Replace the tradable set only when new finalists exist.
                # An empty score must not wipe last quarter's pairs.
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
                    logger.info(
                        "No new finalists on %s — keeping %d existing active pair(s)",
                        today_date,
                        len(active_pairs),
                    )
                    
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
        # Sorted for run-to-run reproducibility: set order varies with hash
        # randomization, and processing order decides which pair wins
        # capacity / shared-ticker priority on days with competing entries.
        pairs_to_process = sorted(set(active_pairs.keys()).union(open_pairs))

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
                
                # ── Dynamic Beta Rebalancing (off by default) ──
                # Resize the B leg when live β drifts. Do not change the
                # formation β/μ/σ used for z-score signals. Disabled via
                # rebalance_beta_intra_trade: the noisy 60-day β realized cash
                # buy-high/sell-low and could flip the short leg's sign (I5).
                if config.rebalance_beta_intra_trade:
                    current_beta = _compute_live_beta(
                        ticker_a, ticker_b, prices_to_date, config
                    )
                    hedge_beta = open_pos.beta_hedge or open_pos.beta_at_entry
                    if current_beta is not None and abs(current_beta - hedge_beta) > config.beta_rebalance_threshold:
                        logger.info(
                            "Beta rebalance %s/%s: hedge %.3f -> %.3f (formation β=%.3f locked)",
                            ticker_a,
                            ticker_b,
                            hedge_beta,
                            current_beta,
                            open_pos.beta_at_entry,
                        )
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

            # ── Per-pair dollar loss cap ─────────────────────────────────────
            # Bounds the loss tail: z-stops let losers run to multiples of the
            # average win (diagnostics I1/I4). Checked before the z-signal so
            # a breached cap closes even when z is still inside the stop band.
            if open_pos is not None and config.max_pair_loss_pct is not None:
                pa = price_today_map.get(ticker_a)
                pb = price_today_map.get(ticker_b)
                if pa is not None and pb is not None:
                    unrealized = (
                        open_pos.shares_a * pa
                        + open_pos.shares_b * pb
                        + open_pos.cumulative_cash_adjustments
                        - open_pos.entry_net_cash_flow
                    )
                    loss_cap = config.max_pair_loss_pct * open_pos.dollar_allocation_at_entry
                    if unrealized <= -loss_cap:
                        logger.info(
                            "DOLLAR_STOP %s/%s — unrealized $%.0f breaches -$%.0f cap",
                            ticker_a, ticker_b, unrealized, loss_cap,
                        )
                        _close_pair(
                            portfolio, ticker_a, ticker_b, pa, pb,
                            "DOLLAR_STOP", trade_log_rows, today_date, prices_to_date,
                        )
                        stoploss_cooldown[(ticker_a, ticker_b)] = config.pair_stop_cooldown_days
                        continue

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

            # ── Exit signals (z/time; DOLLAR_STOP already handled above) ──────
            if signal in ("TAKE_PROFIT", "STOP_LOSS", "TIME_STOP") and open_pos:
                _close_pair(
                    portfolio, ticker_a, ticker_b, price_a, price_b,
                    signal, trade_log_rows, today_date, prices_to_date,
                )
                if signal == "STOP_LOSS":
                    stoploss_cooldown[(ticker_a, ticker_b)] = config.pair_stop_cooldown_days
                continue

            # ── Blocked-entry attribution for HOLD at stretched z ─────────────
            # get_signal returns HOLD both for |z| inside the entry band and
            # when the momentum filter / cross requirement suppressed an entry.
            # Record the latter so filter binding is measurable from data.
            if signal == "HOLD" and open_pos is None and pair in active_pairs:
                _attribute_hold_block(
                    blocked_rows, ticker_a, ticker_b, prices_to_date, today_date,
                    beta_formation, mean_formation, std_formation, config,
                )

            # ── Entry signals ─────────────────────────────────────────────────
            if signal in ("LONG_SPREAD", "SHORT_SPREAD") and open_pos is None:
                if not entries_ok:
                    logger.info("Entry blocked by drawdown halt: %s/%s", ticker_a, ticker_b)
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "drawdown_halt")
                    continue
                if not vix_ok:
                    logger.info("Entry blocked by VIX filter: %s/%s", ticker_a, ticker_b)
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "vix")
                    continue
                if in_blackout(ticker_a, today_date) or in_blackout(ticker_b, today_date):
                    logger.info(
                        "Entry blocked by earnings blackout: %s/%s", ticker_a, ticker_b
                    )
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "earnings_blackout")
                    continue

                # Only open if it's currently an active pair
                if (ticker_a, ticker_b) not in active_pairs:
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "not_active")
                    continue
                if stoploss_cooldown.get((ticker_a, ticker_b), 0) > 0:
                    logger.info(
                        "Entry blocked by stop-loss cooldown: %s/%s (%d day(s) left)",
                        ticker_a,
                        ticker_b,
                        stoploss_cooldown[(ticker_a, ticker_b)],
                    )
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "cooldown")
                    continue

                if not portfolio.can_open(ticker_a, ticker_b, len(active_pairs)):
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "capacity")
                    continue

                beta = beta_formation
                mean = mean_formation
                std  = std_formation

                if beta <= config.min_formation_beta:
                    logger.info(
                        "Entry skipped %s/%s — formation β=%.3f <= %.3f",
                        ticker_a,
                        ticker_b,
                        beta,
                        config.min_formation_beta,
                    )
                    _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "beta")
                    continue

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
                        _record_block(blocked_rows, today_date, ticker_a, ticker_b, signal, "cost_gate")
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
    blocked_entries = (
        pd.DataFrame(blocked_rows)
        if blocked_rows
        else pd.DataFrame(columns=["date", "ticker_a", "ticker_b", "signal", "reason"])
    )

    logger.info(
        "Backtest complete — %d trades, final NAV $%.0f, %d blocked-entry events",
        len(trade_log),
        nav_series["nav"].iloc[-1] if not nav_series.empty else 0,
        len(blocked_entries),
    )

    return trade_log, nav_series, pair_daily, blocked_entries


# ── Helper functions ──────────────────────────────────────────────────────────


def _empty_backtest_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build empty trade log, NAV, pair-daily, and blocked-entry frames.

    Returns:
        Tuple of four empty DataFrames matching ``run_backtest`` outputs.
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
    blocked_entries = pd.DataFrame(
        columns=["date", "ticker_a", "ticker_b", "signal", "reason"]
    )
    return trade_log, nav_series, pair_daily, blocked_entries


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
    # Per-day multiplicative step so the position reaches drawdown_trim_factor
    # of its pre-halt size after drawdown_trim_days days. The old code applied
    # the full factor every day (0.25^5 ≈ 0.1% — a forced liquidation at the
    # drawdown low; see diagnostics RC2).
    daily_factor = config.drawdown_trim_factor ** (1.0 / max(1, config.drawdown_trim_days))
    for pos in list(portfolio.positions.values()):
        target_shares_a = pos.shares_a * daily_factor
        target_shares_b = pos.shares_b * daily_factor
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
            "Trim %s/%s — reduced to %.1f%% of yesterday's size (targeting %.0f%% overall)",
            pos.ticker_a, pos.ticker_b, daily_factor * 100,
            config.drawdown_trim_factor * 100,
        )


def _record_block(
    blocked_rows: list,
    today_date: date,
    ticker_a: str,
    ticker_b: str,
    signal: str,
    reason: str,
) -> None:
    """
    Append one blocked-entry attribution row.

    Args:
        blocked_rows: Mutable list of row dicts appended in-place.
        today_date: Simulation date the entry was suppressed.
        ticker_a: Leg A ticker.
        ticker_b: Leg B ticker.
        signal: The actionable signal that was blocked (LONG/SHORT_SPREAD).
        reason: Block reason slug (see run_backtest docstring).

    Returns:
        None.
    """
    blocked_rows.append({
        "date": today_date,
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "signal": signal,
        "reason": reason,
    })


def _attribute_hold_block(
    blocked_rows: list,
    ticker_a: str,
    ticker_b: str,
    prices_to_date: pd.DataFrame,
    today_date: date,
    beta_formation: float,
    mean_formation: float,
    std_formation: float,
    config: StrategyConfig,
) -> None:
    """
    Attribute a HOLD on a flat active pair whose |z| is beyond the entry band.

    ``get_signal`` returns HOLD both when z is inside the band and when the
    momentum filter or fresh-cross requirement suppressed an entry. This
    recomputes z and records ``momentum`` or ``no_cross`` so filter binding is
    measurable from ``blocked_entries.csv``. Instrumentation only — no effect
    on trading.

    Args:
        blocked_rows: Mutable list of row dicts appended in-place.
        ticker_a: Leg A ticker.
        ticker_b: Leg B ticker.
        prices_to_date: Prices through the simulation day.
        today_date: Simulation date.
        beta_formation: Locked formation hedge ratio.
        mean_formation: Locked formation spread mean.
        std_formation: Locked formation spread std.
        config: Strategy parameters.

    Returns:
        None.
    """
    _, z_score = compute_spread(
        ticker_a, ticker_b, beta_formation, mean_formation, std_formation,
        prices_to_date, today_date, config=config,
    )
    if z_score != z_score or abs(z_score) < config.entry_zscore:
        return

    side = "LONG_SPREAD" if z_score <= -config.entry_zscore else "SHORT_SPREAD"
    if _is_momentum_breakout(ticker_a, ticker_b, prices_to_date, today_date, config=config):
        reason = "momentum"
    else:
        # Only remaining way get_signal held a stretched flat pair.
        reason = "no_cross"
    _record_block(blocked_rows, today_date, ticker_a, ticker_b, side, reason)


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

    Updates ``pos.beta_hedge`` and ``pos.shares_b`` only. Formation
    ``beta_at_entry`` / ``mean_at_entry`` / ``std_at_entry`` stay locked
    so z-score signals are not poisoned.

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
    hedge_beta = pos.beta_hedge if pos.beta_hedge else pos.beta_at_entry
    target_shares_b = pos.shares_b * (new_beta / hedge_beta) if hedge_beta != 0 else 0
    
    if target_shares_b == 0:
        return
        
    # Calculate difference
    diff_shares = target_shares_b - old_shares_b
    cash_impact = -diff_shares * price_b  # Buying shares costs cash, selling adds cash
    
    portfolio.cash += cash_impact
    pos.cumulative_cash_adjustments += cash_impact
    pos.shares_b = target_shares_b
    pos.beta_hedge = new_beta


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