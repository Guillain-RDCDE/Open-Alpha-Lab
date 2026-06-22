"""Data layer for Study 352 — Opening-Range Breakout (ORB).

Two sources, both offline-friendly:

* **Real tape.** 5-minute intraday bars for an ETF (default ``QQQ``; ``SPY`` also cached),
  pulled once from **yfinance** (Yahoo public endpoint, no key) and cached under
  ``_cache/<sym>_5m.csv``. yfinance only serves ~60 calendar days of 5-minute history, so the
  real sample is intentionally short (~60 trading days) — we say so loudly on every axis and
  lean on the deterministic synthetic control for the machinery proof. We fold the bars into
  **regular-session trading days** (09:30–16:00 ET), keeping the per-bar open/high/low/close so
  the strategy and its random-entry null see *exactly* what an intraday trader would.
* **Synthetic.** A deterministic per-minute arithmetic-Brownian day generator (fixed seed) used
  as a *positive control*. It plants a known intraday trend-persistence (an AR(1) drift in the
  per-bar steps) so the ORB edge is real **by construction** — the harness must recover a
  positive, significant ORB-minus-random spread on this tape, or the measurement engine is
  broken. With the persistence switched off (a pure random walk) the same engine must return a
  spread indistinguishable from zero. Both halves are checked in ``examples/verify.py``.

Pure numpy + pandas + stdlib for the offline core; the network is touched *only* on a cache
miss in :func:`fetch_real` (never imported by the synthetic path or the notebooks' frozen
fallback).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# A regular US equity session is 09:30–16:00 ET = 78 five-minute bars.
BARS_PER_DAY = 78
OPEN_HHMM = (9, 30)


def cache_path(symbol: str = "QQQ") -> str:
    return os.path.join(CACHE_DIR, f"{symbol.upper()}_5m.csv")


# --------------------------------------------------------------------------- #
# Real tape — fetch (network, cache-miss only) + load (offline)
# --------------------------------------------------------------------------- #
def fetch_real(symbol: str = "QQQ", period: str = "60d", interval: str = "5m") -> str:
    """Download 5-minute bars from yfinance and cache them as a flat CSV. Network call.

    Only invoked on an explicit cache miss; the offline core never calls this.
    """
    import yfinance as yf  # local import: keep the offline path import-free

    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].copy()
    df.index.name = "datetime"
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(symbol)
    df.to_csv(path)
    return path


def load_bars(symbol: str = "QQQ") -> pd.DataFrame:
    """Cached 5-minute bars as a tz-aware DataFrame indexed by ET timestamp."""
    path = cache_path(symbol)
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # Keep regular-session bars only (drop any pre/post-market rows yfinance may include).
    et = df.index.tz_convert("America/New_York") if df.index.tz is not None else df.index
    df = df.assign(_h=et.hour, _m=et.minute, _d=et.normalize())
    df = df[((df["_h"] > 9) | ((df["_h"] == 9) & (df["_m"] >= 30))) & (df["_h"] < 16)]
    return df


def build_days(symbol: str = "QQQ", min_bars: int = 70) -> np.ndarray:
    """Fold cached 5-minute bars into a 3-D array of complete regular sessions.

    Returns ``days`` with shape ``(n_days, BARS_PER_DAY, 4)``; the last axis is
    ``[open, high, low, close]`` per bar, padded/truncated to ``BARS_PER_DAY`` bars and
    keeping only days with at least ``min_bars`` real bars. Deterministic given the cache.
    """
    return _fold_days(load_bars(symbol), min_bars=min_bars)


def _fold_days(df: pd.DataFrame, min_bars: int = 70) -> np.ndarray:
    out = []
    for _, g in df.groupby(df["_d"]):
        g = g.sort_index()
        if len(g) < min_bars:
            continue
        ohlc = g[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
        if len(ohlc) >= BARS_PER_DAY:
            ohlc = ohlc[:BARS_PER_DAY]
        else:  # pad the tail by repeating the last bar (a flat hold to the close)
            pad = np.repeat(ohlc[-1:], BARS_PER_DAY - len(ohlc), axis=0)
            ohlc = np.vstack([ohlc, pad])
        out.append(ohlc)
    return np.asarray(out, dtype=float)


def have_real(symbol: str = "QQQ") -> bool:
    return os.path.exists(cache_path(symbol))


# --------------------------------------------------------------------------- #
# Synthetic positive control — a deterministic day generator with planted trend
# --------------------------------------------------------------------------- #
def synthetic_days(n_days: int = 400, persistence: float = 0.18,
                   sigma_per_bar: float = 0.12, start_px: float = 500.0,
                   bars: int = BARS_PER_DAY, seed: int = 352) -> np.ndarray:
    """Deterministic intraday sessions with a *known* trend-persistence (positive control).

    Each bar's log-return is an AR(1): ``r_t = persistence * r_{t-1} + eps_t`` with
    ``eps ~ N(0, sigma_per_bar%)``. A positive ``persistence`` means a move that breaks the
    opening range *tends to continue* — exactly the mechanism the ORB monetises — so the ORB
    strategy must beat random entries on this tape. Set ``persistence=0`` for a pure random
    walk, where ORB must *not* beat random. Returns ``(n_days, bars, 4)`` ``[o,h,l,c]``; the
    high/low straddle each bar's open/close by a small fixed wick so range breaks are testable.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma_per_bar / 100.0, size=(n_days, bars))
    r = np.empty_like(eps)
    r[:, 0] = eps[:, 0]
    for t in range(1, bars):
        r[:, t] = persistence * r[:, t - 1] + eps[:, t]
    # bar closes from compounded returns; opens are the previous close
    logpx = np.cumsum(r, axis=1)
    close = start_px * np.exp(logpx)
    open_ = np.empty_like(close)
    open_[:, 0] = start_px
    open_[:, 1:] = close[:, :-1]
    wick = sigma_per_bar / 100.0 * 0.5
    hi = np.maximum(open_, close) * (1.0 + wick)
    lo = np.minimum(open_, close) * (1.0 - wick)
    return np.stack([open_, hi, lo, close], axis=2)
