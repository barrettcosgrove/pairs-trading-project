# Tracks open positions, per-pair P&L, and portfolio NAV.
# Implements dollar-neutral sizing, dynamic equal capital allocation across
# active pairs, β-rebalancing, and drawdown controls.
# Enforces the no-shared-stocks constraint across all pairs.

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import CONFIG, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single open pairs position."""
    ticker_a: str
    ticker_b: str
    cluster_id: int
    direction: str               # "LONG_SPREAD" or "SHORT_SPREAD"
    shares_a: float              # positive = long, negative = short
    shares_b: float              # positive = long, negative = short
    beta_at_entry: float
    mean_at_entry: float
    std_at_entry: float
    expected_halflife: float
    entry_net_cash_flow: float
    dollar_allocation_at_entry: float
    entry_transaction_cost: float
    entry_date: date
    # Live hedge ratio used only to size/rebalance shares. Signals always use
    # beta_at_entry / mean_at_entry / std_at_entry (locked at formation).
    beta_hedge: float = 0.0
    cumulative_cash_adjustments: float = 0.0
    days_open: int = 0
    realized_pnl: float = 0.0


class Portfolio:
    """
    Manages all open positions and portfolio NAV.

    Enforces dollar-neutral sizing, max pairs cap, no-shared-stocks
    constraint, and drawdown controls throughout the backtest simulation.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """
        Initialize portfolio with starting capital from ``config``.

        Cash starts at ``config.initial_capital``. Each pair receives
        equal_weight of NAV, with ``config.cash_buffer_pct`` held in reserve.
        Defaults to global ``CONFIG`` when ``config`` is None.

        Args:
            config: Strategy parameters for sizing and drawdown rules; defaults
                to ``CONFIG`` so standalone tests keep prior behavior.
        """
        self._cfg = config or CONFIG
        ic = float(self._cfg.initial_capital)
        self.cash: float = ic
        self.positions: dict[tuple[str, str], Position] = {}
        self.peak_nav: float = ic
        self.prior_week_nav: float = ic
        self.nav_history: list[tuple[date, float]] = []
        self._halted: bool = False
        self._trim_days_remaining: int = 0
        self._peak_reset_done: bool = False
        self._recovery_days_count: int = 0
        self._recovery_ref_nav: float | None = None

    # ── NAV and capital ───────────────────────────────────────────────────────

    def compute_nav(self, prices: pd.DataFrame, as_of: date) -> float:
        """
        Compute current portfolio NAV from open positions and cash.

        Args:
            prices: DataFrame with columns [date, ticker, adj_close].
            as_of: Valuation date.

        Returns:
            Total NAV in USD.
        """
        price_map = (
            prices[prices["date"] == pd.Timestamp(as_of)]
            .set_index("ticker")["adj_close"]
            .to_dict()
        )

        position_value = 0.0
        for pos in self.positions.values():
            price_a = price_map.get(pos.ticker_a, 0.0)
            price_b = price_map.get(pos.ticker_b, 0.0)
            position_value += pos.shares_a * price_a + pos.shares_b * price_b

        nav = self.cash + position_value
        self.nav_history.append((as_of, nav))
        self._update_peak(nav)
        return nav

    def _update_peak(self, nav: float) -> None:
        """Update rolling peak NAV for drawdown computation."""
        if nav > self.peak_nav:
            self.peak_nav = nav

    def drawdown_from_peak(self, nav: float) -> float:
        """
        Compute current drawdown from rolling peak NAV.

        Args:
            nav: Current portfolio NAV.

        Returns:
            Drawdown as a positive fraction (e.g. 0.08 = 8% drawdown).
        """
        if self.peak_nav <= 0:
            return 0.0
        return max(0.0, (self.peak_nav - nav) / self.peak_nav)

    def drawdown_from_prior_week(self, nav: float) -> float:
        """
        Compute drawdown from the prior week's closing NAV.

        Args:
            nav: Current portfolio NAV.

        Returns:
            Drawdown as a positive fraction.
        """
        if self.prior_week_nav <= 0:
            return 0.0
        return max(0.0, (self.prior_week_nav - nav) / self.prior_week_nav)

    def update_prior_week_nav(self, nav: float) -> None:
        """
        Store NAV as the prior week reference. Call once per week in engine.

        Args:
            nav: End-of-week NAV to record.
        """
        self.prior_week_nav = nav

    # ── Drawdown controls ─────────────────────────────────────────────────────

    def check_drawdown_controls(self, nav: float) -> tuple[bool, float]:
        """
        Evaluate drawdown thresholds and return entry permission and size factor.

        Soft threshold (5% from prior week): new entries at 50% normal size.
        Hard threshold (10% from rolling peak): halt all new entries and trim
        existing positions to ``drawdown_trim_factor`` over
        ``drawdown_trim_days``. Once the trim completes, the rolling peak is
        reset to current NAV and the halt releases after
        ``drawdown_recovery_days`` consecutive non-losing days — a temporary
        circuit breaker, not a permanent stop. (The old release rule required
        NAV within 5% of the all-time peak with no entries allowed, which
        deadlocked the book permanently from 2023-03-09.)

        Args:
            nav: Current portfolio NAV.

        Returns:
            Tuple of (new_entries_permitted, size_factor) where:
                new_entries_permitted — False if hard halt is active.
                size_factor           — 1.0 normal, 0.5 soft reduction.
        """
        dd_peak = self.drawdown_from_peak(nav)
        dd_week = self.drawdown_from_prior_week(nav)

        if self._halted:
            if self._trim_days_remaining > 0:
                return False, 0.0

            if not self._peak_reset_done:
                # Trim finished: re-anchor the drawdown baseline so recovery is
                # measured from the post-trim book, not the pre-crash peak.
                logger.info(
                    "Drawdown trim complete — resetting peak NAV $%.0f -> $%.0f "
                    "and starting %d-day recovery count.",
                    self.peak_nav, nav, self._cfg.drawdown_recovery_days,
                )
                self.peak_nav = nav
                self._peak_reset_done = True
                self._recovery_days_count = 0
                self._recovery_ref_nav = nav
                return False, 0.0

            if self._recovery_ref_nav is None or nav >= self._recovery_ref_nav:
                self._recovery_days_count += 1
            else:
                self._recovery_days_count = 0
            self._recovery_ref_nav = nav

            if self._recovery_days_count >= self._cfg.drawdown_recovery_days:
                logger.info(
                    "Drawdown recovery condition met (%d non-losing days) — "
                    "resuming normal operations at NAV $%.0f.",
                    self._recovery_days_count, nav,
                )
                self._halted = False
                self._peak_reset_done = False
                self._recovery_days_count = 0
                self._recovery_ref_nav = None
            else:
                return False, 0.0

        elif dd_peak >= self._cfg.drawdown_halt_threshold:
            logger.warning(
                "HARD HALT — drawdown from peak %.1f%% exceeds %.1f%% threshold. "
                "Halting new entries and beginning position trim.",
                dd_peak * 100, self._cfg.drawdown_halt_threshold * 100,
            )
            self._halted = True
            self._trim_days_remaining = self._cfg.drawdown_trim_days
            self._peak_reset_done = False
            self._recovery_days_count = 0
            self._recovery_ref_nav = None
            return False, 0.0

        if dd_week >= self._cfg.drawdown_reduce_threshold:
            logger.info(
                "SOFT REDUCE — weekly drawdown %.1f%% exceeds %.1f%% threshold. "
                "New entries at 50%% size.",
                dd_week * 100, self._cfg.drawdown_reduce_threshold * 100,
            )
            return True, self._cfg.drawdown_reduce_factor

        return True, 1.0

    def needs_trim(self) -> bool:
        """
        Return True if gradual position trim is in progress after hard halt.

        Returns:
            True if trim days remain, False otherwise.
        """
        return self._trim_days_remaining > 0

    def decrement_trim_day(self) -> None:
        """Advance the trim countdown by one trading day."""
        if self._trim_days_remaining > 0:
            self._trim_days_remaining -= 1

    # ── Constraint checks ─────────────────────────────────────────────────────

    def tickers_in_use(self) -> set[str]:
        """
        Return the set of all tickers currently in an open position.

        Returns:
            Set of ticker strings.
        """
        tickers: set[str] = set()
        for pos in self.positions.values():
            tickers.add(pos.ticker_a)
            tickers.add(pos.ticker_b)
        return tickers

    def can_open(
        self,
        ticker_a: str,
        ticker_b: str,
        active_pairs_count: int,
    ) -> bool:
        """
        Check all constraints before opening a new position.

        Enforces: portfolio pair cap (active_pairs_count), no shared tickers,
        and gross leverage cap.

        Args:
            ticker_a: Ticker for leg A.
            ticker_b: Ticker for leg B.
            active_pairs_count: Total number of active pairs generated by scoring.

        Returns:
            True if all constraints pass, False otherwise.
        """
        # Portfolio pair cap tied to active pairs count
        if len(self.positions) >= active_pairs_count:
            logger.info(
                "Rejecting %s/%s — portfolio at capacity (%d pairs)",
                ticker_a, ticker_b, active_pairs_count,
            )
            return False

        # No shared tickers
        active_tickers = self.tickers_in_use()
        if ticker_a in active_tickers or ticker_b in active_tickers:
            logger.info(
                "Rejecting %s/%s — ticker already in an open position",
                ticker_a, ticker_b,
            )
            return False

        return True

    def position_size(self, nav: float, size_factor: float, active_pairs_count: int) -> float:
        """
        Compute the dollar allocation for one leg of a new position.

        Splits investable capital across ``target_concurrent_pairs`` expected
        simultaneous positions (or fewer if fewer pairs are active), scaled by
        the drawdown size factor and capped at ``max_weight_per_pair`` of NAV.
        Dividing by the full active-pair count sized for a concurrency the book
        never reached (median 2 open pairs vs ~10 active), stranding capital.

        Args:
            nav: Current portfolio NAV.
            size_factor: Drawdown size multiplier (1.0 normal, 0.5 soft reduce).
            active_pairs_count: Number of pairs currently scored as tradable.

        Returns:
            Dollar value to allocate to each leg of the new position.
        """
        investable_capital = nav * (1.0 - self._cfg.cash_buffer_pct)
        divisor = max(1, min(active_pairs_count, self._cfg.target_concurrent_pairs))
        equal_weight_alloc = investable_capital / divisor
        max_alloc = nav * self._cfg.max_weight_per_pair
        return min(equal_weight_alloc, max_alloc) * size_factor

    # ── Open and close ────────────────────────────────────────────────────────

    def open_position(
        self,
        ticker_a: str,
        ticker_b: str,
        cluster_id: int,
        direction: str,
        beta: float,
        mean: float,
        std: float,
        expected_halflife: float,
        price_a: float,
        price_b: float,
        dollar_allocation: float,
        entry_date: date,
        entry_transaction_cost: float = 0.0,
    ) -> Position | None:
        """
        Open a new dollar-neutral position and deduct cost from cash.

        Long leg receives dollar_allocation; short leg is sized at
        dollar_allocation × beta to maintain dollar neutrality.

        Args:
            ticker_a: Ticker for leg A (dependent).
            ticker_b: Ticker for leg B (independent).
            cluster_id: K-means cluster id.
            direction: "LONG_SPREAD" (long A, short B) or "SHORT_SPREAD".
            beta: Locked hedge ratio from formation period.
            mean: Locked spread mean from formation period.
            std: Locked spread std from formation period.
            expected_halflife: The expected halflife to revert.
            price_a: Fill price for ticker A.
            price_b: Fill price for ticker B.
            dollar_allocation: Dollar value to assign to the long leg.
            entry_date: Trade date.
            entry_transaction_cost: Commission and half-spread cost paid on open (USD).

        Returns:
            Opened Position, or None if cash is insufficient.
        """
        shares_a_raw = dollar_allocation / price_a
        shares_b_raw = (dollar_allocation * beta) / price_b

        if direction == "LONG_SPREAD":
            shares_a = shares_a_raw
            shares_b = -shares_b_raw
        else:
            shares_a = -shares_a_raw
            shares_b = shares_b_raw

        # In a dollar-neutral pair, shares_a * price_a + shares_b * price_b is roughly 0
        # For a long spread, shares_a is positive (we pay cash), shares_b is negative (we receive cash)
        net_cash_flow = shares_a * price_a + shares_b * price_b
        
        # We don't block on "insufficient cash" because short proceeds fund the long leg.
        # Position caps and leverage limits are handled by can_open() and position_size().
        
        self.cash -= net_cash_flow
        pos = Position(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            cluster_id=cluster_id,
            direction=direction,
            shares_a=shares_a,
            shares_b=shares_b,
            beta_at_entry=beta,
            mean_at_entry=mean,
            std_at_entry=std,
            beta_hedge=beta,
            expected_halflife=expected_halflife,
            entry_net_cash_flow=net_cash_flow,
            dollar_allocation_at_entry=float(dollar_allocation),
            entry_transaction_cost=float(entry_transaction_cost),
            entry_date=entry_date,
        )
        self.positions[(ticker_a, ticker_b)] = pos
        logger.info(
            "Opened %s %s/%s — $%.0f allocation, β=%.3f, shares_a=%.2f, shares_b=%.2f",
            direction, ticker_a, ticker_b, dollar_allocation, beta, shares_a, shares_b,
        )
        return pos

    def close_position(
        self,
        ticker_a: str,
        ticker_b: str,
        price_a: float,
        price_b: float,
        reason: str,
    ) -> float:
        """
        Close an open position and return cash proceeds.

        Args:
            ticker_a: Ticker for leg A.
            ticker_b: Ticker for leg B.
            price_a: Exit fill price for ticker A.
            price_b: Exit fill price for ticker B.
            reason: Exit reason string for logging (e.g. "TAKE_PROFIT").

        Returns:
            Realized P&L for this position in USD.
        """
        pair = (ticker_a, ticker_b)
        pos = self.positions.get(pair)
        if pos is None:
            logger.warning(
                "Attempted to close non-existent position %s/%s", ticker_a, ticker_b
            )
            return 0.0

        # When closing, we sell the long leg (receive cash) and buy back the short leg (pay cash)
        # pos.shares_a is positive for long, so selling it gives us positive cash.
        net_cash_flow = pos.shares_a * price_a + pos.shares_b * price_b
        self.cash += net_cash_flow
        
        # Realized spread P&L follows full lifecycle cash accounting:
        # entry outflow/inflow + any intra-life cash adjustments (rebalance/trim)
        # + exit proceeds. Costs are layered by the engine.
        pnl = (
            net_cash_flow
            + pos.cumulative_cash_adjustments
            - pos.entry_net_cash_flow
        )

        logger.info(
            "Closed %s/%s (%s) — reason=%s, P&L=$%.2f",
            ticker_a, ticker_b, pos.direction, reason, pnl,
        )
        del self.positions[pair]
        return pnl

    def increment_days_open(self) -> None:
        """Increment days_open counter for all open positions. Call once per day."""
        for pos in self.positions.values():
            pos.days_open += 1