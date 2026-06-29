# Portfolio-level regime filter. Blocks all new position entries when VIX
# exceeds CONFIG.vix_entry_block. Resumes only after VIX stays below
# CONFIG.vix_resume for CONFIG.vix_resume_days consecutive trading days.

import logging
from datetime import date

import pandas as pd

from src.config import CONFIG

logger = logging.getLogger(__name__)


def new_entries_permitted(as_of: date, vix_series: pd.Series) -> bool:
    """
    Determine whether new position entries are permitted on a given date.

    Blocks all new entries when VIX exceeds CONFIG.vix_entry_block (28.0).
    Once blocked, entries only resume after VIX has stayed below
    CONFIG.vix_resume (25.0) for CONFIG.vix_resume_days (5) consecutive
    trading days. This prevents whipsawing back into positions during
    brief VIX dips within a volatile period.

    Args:
        as_of: The date to evaluate. Only VIX data on or before this date
               is used — no look-ahead bias.
        vix_series: Daily VIX close series indexed by date, as returned
                    by load_vix(). Name should be "vix".

    Returns:
        True if new entries are permitted on as_of, False if blocked.
    """
    vix_to_date = vix_series[vix_series.index <= pd.Timestamp(as_of)]

    if vix_to_date.empty:
        logger.warning("No VIX data available as of %s — blocking entries by default", as_of)
        return False

    vix_today = vix_to_date.iloc[-1]

    # ── Immediate block ───────────────────────────────────────────────────────
    if vix_today > CONFIG.vix_entry_block:
        logger.info(
            "New entries BLOCKED — VIX %.1f exceeds threshold %.1f on %s",
            vix_today, CONFIG.vix_entry_block, as_of,
        )
        return False

    # ── Resume condition: must hold below vix_resume for vix_resume_days ─────
    # Check whether VIX was ever above the block threshold recently enough
    # that the resume condition might not yet be satisfied.
    recent = vix_to_date.iloc[-CONFIG.vix_resume_days:]

    if len(recent) < CONFIG.vix_resume_days:
        # Not enough history to confirm the resume window — stay blocked
        logger.info(
            "New entries BLOCKED — insufficient VIX history to confirm resume (%d days, need %d)",
            len(recent), CONFIG.vix_resume_days,
        )
        return False

    # Check whether any day in the resume window was above the block threshold.
    # If VIX was above vix_entry_block within the last vix_resume_days days,
    # the resume condition cannot yet have been satisfied.
    if (recent > CONFIG.vix_entry_block).any():
        logger.info(
            "New entries BLOCKED — VIX exceeded block threshold within last %d days",
            CONFIG.vix_resume_days,
        )
        return False

    # All vix_resume_days days must be below vix_resume, not just vix_entry_block
    if (recent > CONFIG.vix_resume).any():
        logger.info(
            "New entries BLOCKED — VIX has not held below resume threshold %.1f for %d days",
            CONFIG.vix_resume, CONFIG.vix_resume_days,
        )
        return False

    logger.debug("New entries PERMITTED — VIX %.1f on %s", vix_today, as_of)
    return True