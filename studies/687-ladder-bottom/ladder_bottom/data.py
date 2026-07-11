"""Data layer for Study 687 — Ladder Bottom.

The **ladder bottom** is a five-candle bullish reversal from the Japanese candlestick
canon (Nison, *Japanese Candlestick Charting Techniques*): a downtrend produces **four**
consecutive falling (bearish) candles — a visible "ladder" stepping down — and the
**fifth** candle reverses, opening higher and closing bullish. The claim: this specific
five-bar shape marks the end of the downtrend, not just any four-red-days-then-a-green-day
coincidence.

Because it needs *five* independent low-probability events to line up in a row (four
falling candles with strictly descending closes, then a bullish break), it is
combinatorially rarer than the desk's three-candle patterns (morning-star, three-black-
crows). To have any chance of a usable sample we pool a broad, long-listed basket.

This module produces two tapes, both a tz-naive daily OHLCV frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss
  (with retries + backoff), then caches so re-runs are offline. SPY + 60 long-listed,
  liquid US large-caps, ~25 years each. This is a **survivors** basket (all still
  trading) — named on the Signal axis, though for a single-pattern event study it
  affects which *names* contribute events, not the direction of the comparison.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning
  ``(data, truth)``. A single ``edge`` knob plants the only structure a ladder-bottom
  rule could possibly harvest: forced 5-bar ladder blocks (four engineered declining
  bearish candles preceded by a mild downtrend, then an engineered bullish reversal) at
  a controlled rate, each optionally followed by a genuine, signed post-pattern bounce.
  ``edge = 0`` is a pure random walk beyond the forced candle shapes — the ladder blocks
  carry no forward information, and the detector must not manufacture significance from
  them. A synthetic positive control proves the machinery can find a planted edge; it
  never backs a REAL stamp on the real tape.

No look-ahead is baked in here: the ladder bottom is confirmed on bars *t-4..t*'s OHLC;
forward returns are measured strictly from *t+1*'s open onward (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A broad, transparent, fixed basket of long-listed US large-caps with deep, clean OHLC
# histories on yfinance, plus SPY as the index reference — the same basket idiom sibling
# study 685 (tri-star doji) uses, chosen for the same reason: a rare multi-bar pattern
# needs many names, many years, for any usable sample at all. This is a *survivors*
# basket — survivorship is named on the Signal axis.
BASKET = [
    "SPY",
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "GS", "AXP", "BAC",
    "C", "WFC", "MS", "T", "VZ", "NKE", "LOW", "UPS", "FDX", "EMR",
    "ETN", "ITW", "APD", "ECL", "PPG", "NEM", "DE", "GD", "LMT", "NOC",
    "RTX", "ADP", "PAYX", "SYK", "MDT", "BDX", "CL", "KMB", "GIS", "SO",
]

AS_OF = "2026-06-30"  # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"lb_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "25y", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLCV for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline. With
    ``fetch=True`` it downloads (retrying a couple of times with backoff on a transient
    failure) and writes the parquet cache. Returns a tz-naive daily OHLCV frame sliced
    to :data:`AS_OF` (no partial-month drift).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars.dropna(how="any")
    return bars.loc[bars.index <= pd.Timestamp(AS_OF)]


def load_real(cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first dict ``{ticker: daily OHLCV frame}`` for the basket.

    With ``fetch=False`` (default) reads cached parquets only. With ``fetch=True``
    warms the cache from yfinance. Tickers with no usable bars are skipped.
    """
    tickers = tickers or BASKET
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, cache_dir=cache_dir, fetch=fetch)
        except (FileNotFoundError, RuntimeError):
            if not fetch:
                raise
    return out


def have_real(cache_dir: str = DEFAULT_CACHE, tickers: list[str] | None = None) -> bool:
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_names: int = 40, n_days: int = 3500, edge: float = 0.0,
                    seed: int = 687, daily_vol: float = 0.013,
                    ladder_rate: float = 0.004, prior_lookback: int = 10,
                    start: str = "2005-01-03") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-ladder-bottom bounce (knob ``edge``).

    Each name is a daily random walk. On a fraction ``ladder_rate`` of eligible sessions
    we force a **5-bar ladder block**: a mild downtrend drift over the preceding
    ``prior_lookback`` sessions (so the block genuinely sits in a downtrend, matching
    the real detector's context requirement), then four engineered strictly-declining
    bearish returns, then one engineered bullish reversal return — exactly the shape
    :func:`strategy.ladder_bottom_flags` keys on. When ``edge != 0``, each forced block
    injects a **signed drift** into the following ``H=20`` sessions proportional to
    ``edge`` — the genuine "the downtrend ends" bounce the folk claim describes. When
    ``edge = 0`` the ladder blocks carry no forward information beyond the engineered
    shape itself, so a forward-return test on them must NOT manufacture significance
    however the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 20  # horizon over which a planted bounce plays out

    data: dict[str, pd.DataFrame] = {}
    for name in names:
        ret = rng.normal(0.0002, daily_vol, n_days)
        is_block_start = rng.random(n_days) < ladder_rate
        is_forced = np.zeros(n_days, dtype=bool)
        lo_bound = prior_lookback + 5
        hi_bound = n_days - H - 6
        for i in range(lo_bound, hi_bound):
            if not is_block_start[i]:
                continue
            if is_forced[max(0, i - 5):i + 5].any():
                continue  # never overlap two forced blocks
            # a mild downtrend leading into the block (the pattern's own context)
            ret[i - prior_lookback:i] -= 0.0015
            # four engineered, strictly-declining bearish sessions (the "ladder")
            down_mags = rng.uniform(0.006, 0.016, 4)
            for k in range(4):
                ret[i + k] = -down_mags[k]
                is_forced[i + k] = True
            # the fifth session: an engineered bullish reversal
            ret[i + 4] = rng.uniform(0.008, 0.022)
            is_forced[i + 4] = True
            if edge != 0.0:
                ret[i + 5:i + 5 + H] += edge / H
        close = 100.0 * np.exp(np.cumsum(ret))
        open_ = np.empty_like(close)
        open_[0] = 100.0
        open_[1:] = close[:-1]
        wick = np.abs(rng.normal(daily_vol * 0.4, daily_vol * 0.15, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "ladder_rate": ladder_rate, "seed": seed, "horizon": H}
    return data, truth


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


def panel_fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """One fingerprint over the whole basket (concatenated closes, ticker-sorted)."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
