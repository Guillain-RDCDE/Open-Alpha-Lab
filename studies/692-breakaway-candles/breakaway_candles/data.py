"""Data layer for Study 692 — Breakaway Candles.

The **breakaway** is a five-candle reversal from the Japanese candlestick canon (Nison,
*Japanese Candlestick Charting Techniques*): candle 1 is a long-bodied candle continuing
the prevailing trend; candle 2 **gaps** away from it in the same direction (a genuine
window that stays open); candles 3-4 **run** further in that direction (the "breakaway"
legs); candle 5 is a long-bodied candle in the *opposite* direction that closes back
**through the gap** — Nison's own words are that it "closes within the gap area", i.e.
erases the last leg of the run. The folk claim: this exact five-bar shape marks a
reversal of the trend it interrupted, in *either* direction — a **bullish breakaway**
ends a downtrend (long), a **bearish breakaway** ends an uptrend (short).

This module produces two tapes, both a tz-naive daily OHLCV frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss
  (with retries + backoff), then caches so re-runs are offline. SPY + 60 long-listed,
  liquid US large-caps, ~25 years each — the same fixed basket idiom used by sibling
  studies 685-tri-star-doji / 687-ladder-bottom (a rare multi-bar candle shape needs a
  broad basket for any usable sample at all). This is a **survivors** basket (all still
  trading) — named on the Signal axis, though for a single-pattern event study it affects
  which *names* contribute events, not the direction of the comparison.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning
  ``(data, truth)``. A single ``edge`` knob plants the only structure a breakaway rule
  could possibly harvest: forced 5-bar breakaway blocks (engineered long candle 1, a
  genuine gap, two engineered "run" candles, then an engineered long reversal candle
  closing back through the gap), alternating bullish/bearish, at a controlled rate, each
  optionally followed by a genuine, signed post-pattern drift. ``edge = 0`` is a pure
  random walk beyond the forced candle shapes — the breakaway blocks carry no forward
  information, and the detector must not manufacture significance from them. A synthetic
  positive control proves the machinery can find a planted edge; it never backs a REAL
  stamp on the real tape.

No look-ahead is baked in here: a breakaway is confirmed on bars *t-4..t*'s OHLC; forward
returns are measured strictly from *t+1*'s open onward (see ``strategy.py``).
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
# histories on yfinance, plus SPY as the index reference — the same shared basket used by
# sibling studies 685-tri-star-doji / 687-ladder-bottom, chosen for the same reason: a
# rare multi-bar candle shape needs many names, many years, for any usable sample at all.
# This is a *survivors* basket — survivorship is named on the Signal axis.
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
    return os.path.join(cache_dir, f"bwc_{safe}_1d.parquet")


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

    With ``fetch=False`` (default) reads cached parquets only. With ``fetch=True`` warms
    the cache from yfinance. Tickers with no usable bars are skipped.
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
                    seed: int = 692, daily_vol: float = 0.013,
                    breakaway_rate: float = 0.004, lookback: int = 10,
                    start: str = "2005-01-03") -> tuple[dict, dict]:
    """Deterministic OHLCV panel with a PLANTED post-breakaway drift (knob ``edge``).

    Each name is a daily random walk. On a fraction ``breakaway_rate`` of eligible
    sessions we force a **5-bar breakaway block**, alternating bullish/bearish so both
    sides of the claim get planted events: a mild trend drift over the preceding
    ``lookback`` sessions (so the block genuinely sits in the matching trend context),
    then an engineered long candle 1, a genuine gap (candle 2), two engineered "run"
    candles (3-4) continuing the trend, then an engineered long reversal candle (5)
    closing back through the gap — exactly the shape
    :func:`strategy.bullish_breakaway_flags` / :func:`strategy.bearish_breakaway_flags`
    key on. When ``edge != 0`` each forced block injects a **signed drift**, in the
    reversal's direction, into the following ``H=20`` sessions — the genuine "the trend
    ends" bounce the folk claim describes. When ``edge = 0`` the breakaway blocks carry
    no forward information beyond the engineered shape itself, so a forward-return test
    on them must NOT manufacture significance however the noise falls.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLCV frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    names = [f"N{i:02d}" for i in range(n_names)]
    H = 20  # horizon over which a planted drift plays out

    data: dict[str, pd.DataFrame] = {}
    for name in names:
        base_ret = rng.normal(0.0002, daily_vol, n_days)
        is_block_start = rng.random(n_days) < breakaway_rate
        forced = np.zeros(n_days, dtype=bool)
        lo_bound = lookback + 5
        hi_bound = n_days - H - 6
        blocks: list[tuple[int, bool]] = []
        block_i = 0
        for i in range(lo_bound, hi_bound):
            if not is_block_start[i] or forced[max(0, i - 5):i + 6].any():
                continue
            bull = (block_i % 2 == 0)  # alternate bullish/bearish across the tape
            block_i += 1
            blocks.append((i, bull))
            forced[i:i + 5] = True
            sign = 1.0 if bull else -1.0
            # a mild trend leading INTO the block, in the pattern's own direction --
            # applied to the baseline returns of the days before candle 1
            base_ret[i - lookback:i] -= sign * 0.0015
            # a genuine, signed post-reversal drift (the planted "trend ends" bounce)
            if edge != 0.0:
                base_ret[i + 5:i + 5 + H] += sign * edge / H
        block_map = dict(blocks)

        # build OHLC bar-by-bar so a forced block's 5 candles are internally
        # consistent (each candle's open equals the prior candle's close, so the
        # gap/run/reversal geometry the detector reads is exactly what was planted)
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        close[0] = 100.0
        open_[0] = 100.0
        t = 1
        while t < n_days:
            if t in block_map:
                bull = block_map[t]
                sign = 1.0 if bull else -1.0
                i = t
                # candle 1 (i): a long-bodied trend candle (bearish if bull-side,
                # bullish if bear-side)
                open_[i] = close[i - 1]
                body1 = rng.uniform(0.012, 0.020)
                close[i] = open_[i] * (1.0 - sign * body1)
                # candle 2 (i+1): gaps further in the trend direction (the window)
                # and closes a bit further still, without filling the gap
                gap = rng.uniform(0.010, 0.018)
                open_[i + 1] = close[i] * (1.0 - sign * gap)
                body2 = rng.uniform(0.003, 0.008)
                close[i + 1] = open_[i + 1] * (1.0 - sign * body2)
                # candles 3-4 (i+2, i+3): the "run" continues the trend
                open_[i + 2] = close[i + 1]
                body3 = rng.uniform(0.004, 0.010)
                close[i + 2] = open_[i + 2] * (1.0 - sign * body3)
                open_[i + 3] = close[i + 2]
                body4 = rng.uniform(0.004, 0.010)
                close[i + 3] = open_[i + 3] * (1.0 - sign * body4)
                # candle 5 (i+4): a long reversal candle closing back through the gap
                open_[i + 4] = close[i + 3]
                body5 = rng.uniform(0.020, 0.032)
                close[i + 4] = open_[i + 4] * (1.0 + sign * body5)
                t += 5
                continue
            open_[t] = close[t - 1]
            close[t] = open_[t] * (1.0 + base_ret[t])
            t += 1

        wick = np.abs(rng.normal(daily_vol * 0.4, daily_vol * 0.15, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
            index=pd.DatetimeIndex(cal, name="date"),
        )
    truth = {"edge": edge, "n_names": n_names, "n_days": n_days,
             "breakaway_rate": breakaway_rate, "seed": seed, "horizon": H}
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
