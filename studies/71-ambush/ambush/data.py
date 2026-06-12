"""Data layer — SPY split-only OHLC, ^VIX raw closes, ^IRX cash rate.

Offline by default: every loader reads the repo-root ``_cache/`` parquets (absolute
path, so notebooks/ and examples/ resolve the same files) and only touches the
network when ``fetch=True`` is passed explicitly. Data choices, stated:

- **SPY split-only** — the confluence signals are daily intraday-shape and calendar
  signals; dividends are immaterial at a 1–3 night horizon, and split-only keeps the
  genuine ex-div gap a CFD holder actually experiences (study 19's choice, same
  reasoning).
- **^VIX raw** — an index level, never traded here, only thresholded.
- **rf** — the 13-week T-bill (^IRX) as a *per-day* return series, the bench's cash
  convention (shared with study 42's cache); it credits the cash leg and prices the
  CFD financing charge.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# studies/71-ambush/ambush/data.py -> repo root
_REPO = Path(__file__).resolve().parents[3]
CACHE = _REPO / "_cache"

START = "1993-01-29"  # SPY's first session


def _load(ticker: str, mode: str, fetch: bool) -> pd.DataFrame:
    """Read ``_cache/<ticker>_<mode>.parquet``; download (full history) only on fetch."""
    path = CACHE / f"{ticker}_{mode}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    elif fetch:
        from quantlab import data as qdata

        df = qdata._download(ticker, mode)
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"no cache for {ticker} at {path} — run examples/verify.py --fetch once (network)"
        )
    return df.loc[START:].copy()


def spy_frame(fetch: bool = False) -> pd.DataFrame:
    """Daily split-only SPY OHLC from 1993."""
    return _load("SPY", "split_only", fetch)


def vix_series(fetch: bool = False) -> pd.Series:
    """Daily ^VIX closes (raw — an index level, never adjusted, never traded)."""
    return _load("^VIX", "raw", fetch)["Close"].rename("vix")


def rf_series(fetch: bool = False) -> pd.Series:
    """Per-day cash return from the 13-week T-bill (^IRX), bench convention.

    Reuses study 42's cached series (``_cache/last_call_irx.parquet``, column ``rf``);
    on ``fetch`` rebuilds it the same way: ``(1 + ^IRX/100)^(1/252) − 1``.
    """
    path = CACHE / "last_call_irx.parquet"
    if path.exists():
        return pd.read_parquet(path)["rf"]
    if fetch:
        import yfinance as yf

        irx = yf.download("^IRX", period="max", auto_adjust=False, progress=False)["Close"]
        rf = ((1.0 + irx.squeeze() / 100.0) ** (1.0 / 252.0) - 1.0).rename("rf")
        rf.index.name = "date"
        CACHE.mkdir(parents=True, exist_ok=True)
        rf.to_frame().to_parquet(path)
        return rf
    raise FileNotFoundError(f"no cash-rate cache at {path} — run with --fetch once")
