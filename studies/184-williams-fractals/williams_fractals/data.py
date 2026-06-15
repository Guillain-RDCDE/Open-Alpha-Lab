"""Data layer for Study 184 (Williams-Fractals).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``momentum`` knob
  (bar-level AR(1) return persistence) lets us dial the only structure that the fractal
  *breakout* rule can harvest: short-term price continuation after a swing extreme is
  broken.  ``momentum = 0`` is a pure random walk — a fair coin — so tests can assert the
  breakout signal only edges a random-direction control when we deliberately plant
  persistence, and not otherwise.  This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily bars go back
  decades, giving ~2,500+ bars per instrument for a 10-year window — high statistical
  power, fully offline once the cache is warm.

Bill Williams' fractal is a 5-bar pattern: a **bearish fractal** is a bar whose high is
strictly higher than the two bars on each side (a local 5-bar swing high); a **bullish
fractal** is a bar whose low is strictly lower than the two bars on each side (a local
5-bar swing low).  Two framings are tested in ``strategy.py``:

1. **Reversal framing**: fade the swing extreme — short after a bearish fractal, long
   after a bullish fractal.  Requires mean-reversion, which daily price series typically
   lack at a 3–5 day horizon.

2. **Breakout framing**: trade the first close that breaks above a bullish fractal high
   (long) or below a bearish fractal low (short).  Requires momentum / continuation,
   which can exist on daily bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1000,
    momentum: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2015-01-02",
    seed: int = 184,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable AR(1) momentum knob.

    Log returns follow ``r_t = momentum * r_{t-1} + eps_t`` where ``eps_t`` is i.i.d.
    normal with standard deviation ``daily_vol``.  A positive ``momentum`` plants
    short-term price continuation — the only structure the fractal *breakout* rule can
    harvest: once a fractal high/low is broken, price keeps going.

    - ``momentum = 0``   → a martingale; the fractal breakout is a fair coin.
    - ``momentum > 0``   → return persistence the breakout rule can ride.
    - ``momentum < 0``   → mean reversion; the breakout rule is on the wrong side.

    Note: the *reversal* framing (fading the swing extreme) would benefit from
    negative momentum / mean reversion, but the 5-bar confirmation delay means most of
    the reversal move has already happened before the trade opens — the real tape bears
    this out empirically.  Both framings are tested in ``strategy.py``; the synthetic
    control here exercises the breakout framing where the planted effect is cleanest.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = momentum * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "momentum": momentum,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "10y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``).  Yahoo's daily history goes back decades, giving ~2,500
    bars per instrument for the 10-year window — high statistical power, fully offline
    once the cache is warm.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Ensure tz-naive DatetimeIndex for consistent downstream handling.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
