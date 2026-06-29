"""Parquet cache for working_model price panels (optional load, avoid refetch)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceLoadMeta(TypedDict, total=False):
    """Metadata returned with every successful price load (fetch or parquet)."""

    source: Literal["yfinance", "parquet"]
    cache_path: str | None
    request_start: str
    request_end: str
    panel_first: str
    panel_last: str
    n_rows: int
    n_tickers: int
    fetched_at_iso: str | None
    stored_panel_first: str
    stored_panel_last: str
    use_window_start: str | None
    use_window_end: str | None


def default_cache_raw_dir() -> Path:
    """Return ``working_model/cache/raw`` (next to this file)."""
    return Path(__file__).resolve().parent / "cache" / "raw"


def _normalize_dates(start_date: str, end_date: str) -> tuple[str, str]:
    """Normalize request bounds to ISO date strings for keys and metadata."""
    s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    e = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    return s, e


def _cache_key(tickers: list[str], start_date: str, end_date: str) -> str:
    """Build a short stable hash for cache filenames."""
    payload = json.dumps(
        {"tickers": sorted({str(t).upper() for t in tickers}), "start": start_date, "end": end_date},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _paths_for_key(
    cache_raw_dir: Path,
    start_date: str,
    end_date: str,
    key: str,
    *,
    prefix: str = "scrap_prices",
) -> tuple[Path, Path]:
    """Parquet path and sidecar JSON metadata path."""
    safe_start = start_date.replace(":", "")
    safe_end = end_date.replace(":", "")
    base = cache_raw_dir / f"{prefix}_{safe_start}_{safe_end}_{key}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


def _write_meta(path_json: Path, meta: dict[str, Any]) -> None:
    """Write metadata JSON next to parquet."""
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _read_meta(path_json: Path) -> dict[str, Any] | None:
    """Read metadata JSON if present."""
    if not path_json.is_file():
        return None
    with path_json.open(encoding="utf-8") as f:
        return json.load(f)


def slice_working_panel(
    df: pd.DataFrame,
    use_start: str | pd.Timestamp | None,
    use_end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Restrict the price panel to an inclusive date window (after cache load or fetch).

    Does not change parquet files or yfinance request bounds; only subsets rows.

    Args:
        df: Wide price DataFrame with DatetimeIndex.
        use_start: First calendar date to keep (inclusive); None means first row.
        use_end: Last calendar date to keep (inclusive); None means last row.

    Returns:
        Row-sliced copy of ``df``.

    Raises:
        ValueError: If the slice is empty or ``use_start`` is after ``use_end``.
    """
    if df.empty:
        raise ValueError("Cannot slice an empty price panel.")
    lo = pd.Timestamp(df.index.min()) if use_start is None else pd.Timestamp(use_start)
    hi = pd.Timestamp(df.index.max()) if use_end is None else pd.Timestamp(use_end)
    if lo > hi:
        raise ValueError(f"use_start ({lo.date()}) must be <= use_end ({hi.date()}).")
    out = df.loc[lo:hi].copy()
    if out.empty:
        raise ValueError(
            f"No rows in [{lo.date()} .. {hi.date()}]; stored panel is "
            f"{df.index.min().date()} .. {df.index.max().date()}."
        )
    return out


def _apply_working_window(
    df: pd.DataFrame,
    meta: PriceLoadMeta,
    use_start: str | None,
    use_end: str | None,
) -> tuple[pd.DataFrame, PriceLoadMeta]:
    """If ``use_start`` / ``use_end`` are set, slice and refresh panel fields in metadata."""
    if use_start is None and use_end is None:
        return df, meta
    stored_first = pd.Timestamp(df.index.min()).strftime("%Y-%m-%d")
    stored_last = pd.Timestamp(df.index.max()).strftime("%Y-%m-%d")
    sliced = slice_working_panel(df, use_start, use_end)
    u0 = None if use_start is None else pd.Timestamp(use_start).strftime("%Y-%m-%d")
    u1 = None if use_end is None else pd.Timestamp(use_end).strftime("%Y-%m-%d")
    meta = {
        **meta,
        "stored_panel_first": stored_first,
        "stored_panel_last": stored_last,
        "use_window_start": u0,
        "use_window_end": u1,
        "panel_first": pd.Timestamp(sliced.index.min()).strftime("%Y-%m-%d"),
        "panel_last": pd.Timestamp(sliced.index.max()).strftime("%Y-%m-%d"),
        "n_rows": int(len(sliced)),
    }
    return sliced, meta


def load_or_fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
    use_start: str | None = None,
    use_end: str | None = None,
) -> tuple[pd.DataFrame, PriceLoadMeta]:
    """Load adjusted closes from parquet cache when possible; otherwise fetch via yfinance.

    Cache files are keyed by sorted tickers plus request start/end dates. Writes
    ``*.parquet`` and ``*.meta.json`` under ``cache_raw_dir``.

    Args:
        tickers: Ticker symbols (any order; matching uses sorted set for the key).
        start_date: yfinance start (inclusive).
        end_date: yfinance end (exclusive, same convention as yfinance).
        cache_raw_dir: Directory for parquet cache; defaults to ``working_model/cache/raw``.
        force_refresh: If True, ignore cache and overwrite after fetch.
        use_start: Optional inclusive first date of the **working** panel (row subset after load).
        use_end: Optional inclusive last date of the **working** panel. Independent of yfinance bounds.

    Returns:
        Tuple of (wide price DataFrame, metadata describing source and date coverage).
    """
    cache_raw_dir = cache_raw_dir if cache_raw_dir is not None else default_cache_raw_dir()
    req_start, req_end = _normalize_dates(start_date, end_date)
    key = _cache_key(list(tickers), req_start, req_end)
    pq_path, meta_path = _paths_for_key(
        cache_raw_dir, req_start, req_end, key, prefix="scrap_prices"
    )

    if not force_refresh and pq_path.is_file():
        df = pd.read_parquet(pq_path)
        meta_disk = _read_meta(meta_path)
        fetched_at = None
        if meta_disk:
            fetched_at = meta_disk.get("fetched_at_iso")
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        df.index = idx
        meta: PriceLoadMeta = {
            "source": "parquet",
            "cache_path": str(pq_path.resolve()),
            "request_start": req_start,
            "request_end": req_end,
            "panel_first": pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
            "panel_last": pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
            "n_rows": int(len(df)),
            "n_tickers": int(df.shape[1]),
            "fetched_at_iso": fetched_at,
        }
        logger.info(
            "Loaded prices from cache %s (%s .. %s, %s rows)",
            pq_path.name,
            meta["panel_first"],
            meta["panel_last"],
            meta["n_rows"],
        )
        return _apply_working_window(df, meta, use_start, use_end)

    logger.info(
        "Fetching prices from yfinance (%s tickers, %s .. %s)",
        len(tickers),
        req_start,
        req_end,
    )
    data = yf.download(list(tickers), start=req_start, end=req_end, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.ffill().dropna(axis=1)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_body = {
        "request_start": req_start,
        "request_end": req_end,
        "tickers_sorted": sorted({str(t).upper() for t in tickers}),
        "cache_key": key,
        "fetched_at_iso": fetched_at,
        "panel_first": pd.Timestamp(data.index.min()).strftime("%Y-%m-%d"),
        "panel_last": pd.Timestamp(data.index.max()).strftime("%Y-%m-%d"),
        "n_rows": int(len(data)),
        "n_tickers": int(data.shape[1]),
    }
    cache_raw_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(pq_path)
    _write_meta(meta_path, meta_body)

    out_meta: PriceLoadMeta = {
        "source": "yfinance",
        "cache_path": str(pq_path.resolve()),
        "request_start": req_start,
        "request_end": req_end,
        "panel_first": meta_body["panel_first"],
        "panel_last": meta_body["panel_last"],
        "n_rows": meta_body["n_rows"],
        "n_tickers": meta_body["n_tickers"],
        "fetched_at_iso": fetched_at,
    }
    return _apply_working_window(data, out_meta, use_start, use_end)


def load_or_fetch_volumes(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
    use_start: str | None = None,
    use_end: str | None = None,
) -> tuple[pd.DataFrame, PriceLoadMeta]:
    """Load volumes from parquet cache when possible; otherwise fetch via yfinance."""
    cache_raw_dir = cache_raw_dir if cache_raw_dir is not None else default_cache_raw_dir()
    req_start, req_end = _normalize_dates(start_date, end_date)
    key = _cache_key(list(tickers), req_start, req_end)
    pq_path, meta_path = _paths_for_key(
        cache_raw_dir, req_start, req_end, key, prefix="scrap_volumes"
    )

    if not force_refresh and pq_path.is_file():
        df = pd.read_parquet(pq_path)
        meta_disk = _read_meta(meta_path)
        fetched_at = meta_disk.get("fetched_at_iso") if meta_disk else None
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
        meta: PriceLoadMeta = {
            "source": "parquet",
            "cache_path": str(pq_path.resolve()),
            "request_start": req_start,
            "request_end": req_end,
            "panel_first": pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
            "panel_last": pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
            "n_rows": int(len(df)),
            "n_tickers": int(df.shape[1]),
            "fetched_at_iso": fetched_at,
        }
        return _apply_working_window(df, meta, use_start, use_end)

    logger.info(
        "Fetching volumes from yfinance (%s tickers, %s .. %s)",
        len(tickers),
        req_start,
        req_end,
    )
    data = yf.download(list(tickers), start=req_start, end=req_end, progress=False)["Volume"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    data = data.ffill().dropna(axis=1)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_body = {
        "request_start": req_start,
        "request_end": req_end,
        "tickers_sorted": sorted({str(t).upper() for t in tickers}),
        "cache_key": key,
        "fetched_at_iso": fetched_at,
        "panel_first": pd.Timestamp(data.index.min()).strftime("%Y-%m-%d"),
        "panel_last": pd.Timestamp(data.index.max()).strftime("%Y-%m-%d"),
        "n_rows": int(len(data)),
        "n_tickers": int(data.shape[1]),
    }
    cache_raw_dir.mkdir(parents=True, exist_ok=True)
    data.to_parquet(pq_path)
    _write_meta(meta_path, meta_body)

    out_meta: PriceLoadMeta = {
        "source": "yfinance",
        "cache_path": str(pq_path.resolve()),
        "request_start": req_start,
        "request_end": req_end,
        "panel_first": meta_body["panel_first"],
        "panel_last": meta_body["panel_last"],
        "n_rows": meta_body["n_rows"],
        "n_tickers": meta_body["n_tickers"],
        "fetched_at_iso": fetched_at,
    }
    return _apply_working_window(data, out_meta, use_start, use_end)


def slice_working_series(
    s: pd.Series,
    use_start: str | pd.Timestamp | None,
    use_end: str | pd.Timestamp | None,
) -> pd.Series:
    """Restrict a VIX (or other) series to an inclusive date window after load.

    Args:
        s: Series with DatetimeIndex.
        use_start: First date to keep (inclusive); None keeps from start.
        use_end: Last date to keep (inclusive); None keeps through end.

    Returns:
        Sliced copy of ``s``.

    Raises:
        ValueError: If slice is empty or bounds are invalid.
    """
    if s.empty:
        raise ValueError("Cannot slice an empty series.")
    lo = pd.Timestamp(s.index.min()) if use_start is None else pd.Timestamp(use_start)
    hi = pd.Timestamp(s.index.max()) if use_end is None else pd.Timestamp(use_end)
    if lo > hi:
        raise ValueError(f"use_start ({lo.date()}) must be <= use_end ({hi.date()}).")
    out = s.loc[lo:hi].copy()
    if out.empty:
        raise ValueError(
            f"No VIX rows in [{lo.date()} .. {hi.date()}]; stored series is "
            f"{s.index.min().date()} .. {s.index.max().date()}."
        )
    return out


def load_or_fetch_vix(
    start_date: str,
    end_date: str,
    *,
    vix_ticker: str = "^VIX",
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
    use_start: str | None = None,
    use_end: str | None = None,
) -> tuple[pd.Series, PriceLoadMeta]:
    """Load VIX close from parquet cache or yfinance (single-ticker panel).

    Uses the same request dates and optional working slice as price loads.

    Args:
        start_date: yfinance start (inclusive).
        end_date: yfinance end (exclusive).
        vix_ticker: Symbol for implied volatility index (default ``^VIX``).
        cache_raw_dir: Parquet directory; defaults to ``working_model/cache/raw``.
        force_refresh: If True, refetch and overwrite cache.
        use_start: Optional inclusive first date of working series after load.
        use_end: Optional inclusive last date of working series.

    Returns:
        Tuple of (VIX close series indexed by date, metadata).
    """
    cache_raw_dir = cache_raw_dir if cache_raw_dir is not None else default_cache_raw_dir()
    req_start, req_end = _normalize_dates(start_date, end_date)
    sym = str(vix_ticker).upper()
    key = _cache_key([sym], req_start, req_end)
    pq_path, meta_path = _paths_for_key(
        cache_raw_dir, req_start, req_end, key, prefix="scrap_vix"
    )

    if not force_refresh and pq_path.is_file():
        df = pd.read_parquet(pq_path)
        meta_disk = _read_meta(meta_path)
        fetched_at = meta_disk.get("fetched_at_iso") if meta_disk else None
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
        col = df.columns[0]
        s = df[col].astype(float)
        meta: PriceLoadMeta = {
            "source": "parquet",
            "cache_path": str(pq_path.resolve()),
            "request_start": req_start,
            "request_end": req_end,
            "panel_first": pd.Timestamp(s.index.min()).strftime("%Y-%m-%d"),
            "panel_last": pd.Timestamp(s.index.max()).strftime("%Y-%m-%d"),
            "n_rows": int(len(s)),
            "n_tickers": 1,
            "fetched_at_iso": fetched_at,
        }
        if use_start is not None or use_end is not None:
            stored_first = pd.Timestamp(s.index.min()).strftime("%Y-%m-%d")
            stored_last = pd.Timestamp(s.index.max()).strftime("%Y-%m-%d")
            sliced = slice_working_series(s, use_start, use_end)
            u0 = None if use_start is None else pd.Timestamp(use_start).strftime("%Y-%m-%d")
            u1 = None if use_end is None else pd.Timestamp(use_end).strftime("%Y-%m-%d")
            meta = {
                **meta,
                "stored_panel_first": stored_first,
                "stored_panel_last": stored_last,
                "use_window_start": u0,
                "use_window_end": u1,
                "panel_first": pd.Timestamp(sliced.index.min()).strftime("%Y-%m-%d"),
                "panel_last": pd.Timestamp(sliced.index.max()).strftime("%Y-%m-%d"),
                "n_rows": int(len(sliced)),
            }
            return sliced, meta
        return s, meta

    logger.info("Fetching VIX from yfinance (%s, %s .. %s)", sym, req_start, req_end)
    data = yf.download(sym, start=req_start, end=req_end, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    s = pd.Series(data.astype(float), name="VIX")
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index))
    s = s.ffill()

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_body = {
        "request_start": req_start,
        "request_end": req_end,
        "tickers_sorted": [sym],
        "cache_key": key,
        "fetched_at_iso": fetched_at,
        "panel_first": pd.Timestamp(s.index.min()).strftime("%Y-%m-%d"),
        "panel_last": pd.Timestamp(s.index.max()).strftime("%Y-%m-%d"),
        "n_rows": int(len(s)),
        "n_tickers": 1,
    }
    cache_raw_dir.mkdir(parents=True, exist_ok=True)
    s.to_frame("VIX").to_parquet(pq_path)
    _write_meta(meta_path, meta_body)

    out_meta: PriceLoadMeta = {
        "source": "yfinance",
        "cache_path": str(pq_path.resolve()),
        "request_start": req_start,
        "request_end": req_end,
        "panel_first": meta_body["panel_first"],
        "panel_last": meta_body["panel_last"],
        "n_rows": meta_body["n_rows"],
        "n_tickers": 1,
        "fetched_at_iso": fetched_at,
    }
    if use_start is not None or use_end is not None:
        stored_first = out_meta["panel_first"]
        stored_last = out_meta["panel_last"]
        sliced = slice_working_series(s, use_start, use_end)
        u0 = None if use_start is None else pd.Timestamp(use_start).strftime("%Y-%m-%d")
        u1 = None if use_end is None else pd.Timestamp(use_end).strftime("%Y-%m-%d")
        out_meta = {
            **out_meta,
            "stored_panel_first": stored_first,
            "stored_panel_last": stored_last,
            "use_window_start": u0,
            "use_window_end": u1,
            "panel_first": pd.Timestamp(sliced.index.min()).strftime("%Y-%m-%d"),
            "panel_last": pd.Timestamp(sliced.index.max()).strftime("%Y-%m-%d"),
            "n_rows": int(len(sliced)),
        }
        return sliced, out_meta
    return s, out_meta


def format_price_load_report(meta: PriceLoadMeta) -> str:
    """Format a human-readable one-block summary for console output.

    Args:
        meta: Metadata from ``load_or_fetch_prices``.

    Returns:
        Multi-line string describing source and panel dates.
    """
    src = meta.get("source", "?")
    lines = [
        "--- Price panel ---",
        f"  Source:           {src}",
        f"  Request (yf):     {meta.get('request_start')} .. {meta.get('request_end')} (end exclusive)",
        f"  Panel index:      {meta.get('panel_first')} .. {meta.get('panel_last')} ({meta.get('n_rows')} rows)",
        f"  Tickers:          {meta.get('n_tickers')}",
    ]
    if meta.get("use_window_start") is not None or meta.get("use_window_end") is not None:
        us = meta.get("use_window_start")
        ue = meta.get("use_window_end")
        lines.append(
            f"  Slice (inclusive): {us or '—'} .. {ue or '—'}  (subsets rows after load; cache file unchanged)"
        )
    if meta.get("stored_panel_first") and meta.get("stored_panel_last"):
        if (
            meta.get("stored_panel_first") != meta.get("panel_first")
            or meta.get("stored_panel_last") != meta.get("panel_last")
        ):
            lines.append(
                f"  Stored in cache:  {meta['stored_panel_first']} .. {meta['stored_panel_last']}"
            )
    if meta.get("cache_path"):
        lines.append(f"  Cache file:       {meta['cache_path']}")
    if meta.get("fetched_at_iso"):
        lines.append(f"  Fetched at (UTC): {meta['fetched_at_iso']}")
    return "\n".join(lines)


def format_vix_load_report(meta: PriceLoadMeta) -> str:
    """Format a short summary for VIX cache/fetch metadata.

    Args:
        meta: Metadata from ``load_or_fetch_vix``.

    Returns:
        Multi-line string for console output.
    """
    lines = [
        "--- VIX series ---",
        f"  Source:           {meta.get('source', '?')}",
        f"  Request (yf):     {meta.get('request_start')} .. {meta.get('request_end')} (end exclusive)",
        f"  Panel index:      {meta.get('panel_first')} .. {meta.get('panel_last')} ({meta.get('n_rows')} rows)",
    ]
    if meta.get("use_window_start") is not None or meta.get("use_window_end") is not None:
        lines.append(
            f"  Slice (inclusive): {meta.get('use_window_start') or '—'} .. "
            f"{meta.get('use_window_end') or '—'}"
        )
    if meta.get("cache_path"):
        lines.append(f"  Cache file:       {meta['cache_path']}")
    if meta.get("fetched_at_iso"):
        lines.append(f"  Fetched at (UTC): {meta['fetched_at_iso']}")
    return "\n".join(lines)
