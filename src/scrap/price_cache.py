"""Parquet cache for scrap yfinance price panels (optional load, avoid refetch)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
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


def default_cache_raw_dir() -> Path:
    """Return default directory ``src/scrap/cache/raw`` (next to this package)."""
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


def _paths_for_key(cache_raw_dir: Path, start_date: str, end_date: str, key: str) -> tuple[Path, Path]:
    """Parquet path and sidecar JSON metadata path."""
    safe_start = start_date.replace(":", "")
    safe_end = end_date.replace(":", "")
    base = cache_raw_dir / f"scrap_prices_{safe_start}_{safe_end}_{key}"
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


def load_or_fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    *,
    cache_raw_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, PriceLoadMeta]:
    """Load adjusted closes from parquet cache when possible; otherwise fetch via yfinance.

    Cache files are keyed by sorted tickers plus request start/end dates. Writes
    ``*.parquet`` and ``*.meta.json`` under ``cache_raw_dir``.

    Args:
        tickers: Ticker symbols (any order; matching uses sorted set for the key).
        start_date: yfinance start (inclusive).
        end_date: yfinance end (exclusive, same convention as yfinance).
        cache_raw_dir: Directory for ``cache/raw``; defaults to ``default_cache_raw_dir()``.
        force_refresh: If True, ignore cache and overwrite after fetch.

    Returns:
        Tuple of (wide price DataFrame, metadata describing source and date coverage).
    """
    cache_raw_dir = cache_raw_dir if cache_raw_dir is not None else default_cache_raw_dir()
    req_start, req_end = _normalize_dates(start_date, end_date)
    key = _cache_key(list(tickers), req_start, req_end)
    pq_path, meta_path = _paths_for_key(cache_raw_dir, req_start, req_end, key)

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
        return df, meta

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

    fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    return data, out_meta


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
    if meta.get("cache_path"):
        lines.append(f"  Cache file:       {meta['cache_path']}")
    if meta.get("fetched_at_iso"):
        lines.append(f"  Fetched at (UTC): {meta['fetched_at_iso']}")
    return "\n".join(lines)
