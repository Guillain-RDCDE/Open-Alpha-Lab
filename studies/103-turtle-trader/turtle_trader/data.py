"""Data layer for Study 103 (Turtle-Trader).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A day-level AR(1) drift
  knob (``trend``) lets us dial the only thing the Donchian breakout rule can possibly
  harvest: medium-term return persistence (trending). ``trend=0`` is a random walk —
  no directional predictability — so a test can assert the breakout beats random *only*
  when we plant a trend, and not otherwise. This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network. We pull a long
  history (start=1993-01-01) to capture the pre- and post-publication regimes.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on data up to and including day *t*, positions entered at day *t+1*'s open).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Donchian channel parameters — Turtle System 1 and System 2
SYSTEM1_ENTRY = 20   # 20-day high breakout
SYSTEM1_EXIT  = 10   # 10-day low exit (long) or high exit (short)
SYSTEM2_ENTRY = 55   # 55-day high breakout
SYSTEM2_EXIT  = 20   # 20-day low exit (long) or high exit (short)

# Basket of liquid trend-following instruments
DEFAULT_TICKERS = ["SPY", "GLD", "TLT", "USO", "UUP", "QQQ", "IEF", "DBA"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    trend: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 103,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of medium-term drift.

    Log returns follow a day-level AR(1) with a drift component:
    ``r_t = trend * r_{t-1} + eps_t`` with ``eps_t`` i.i.d. normal scaled to
    ``annual_vol``. The ``trend`` parameter is the *only* forecastable structure
    in the tape:

    - ``trend = 0``   → a martingale; the Donchian breakout is a fair coin.
    - ``trend > 0``   → return persistence (momentum) the breakout rule can ride.
    - ``trend < 0``   → mean reversion the breakout rule is on the *wrong* side of.

    Days are stamped on consecutive business days. Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start=start, periods=n_days)
    daily_vol = annual_vol / np.sqrt(252.0)

    log_ret = np.empty(n_days)
    prev = 0.0
    eps = rng.normal(0.0, daily_vol, n_days)
    for i in range(n_days):
        r = trend * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intrabar high/low: realistic day range as a fraction of daily vol.
    half_range = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + half_range
    lo = np.minimum(open_, close) - half_range
    lo = np.maximum(lo, 1e-6)   # prices stay positive
    vol = rng.integers(500_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=bdays,
    )
    bars.index.name = "date"
    truth = {
        "trend": trend,
        "annual_vol": annual_vol,
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
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Daily history stretches back to 1993 for most ETFs,
    giving a long enough window to split pre-/post-publication regimes (the original
    Turtle program ended around 1988; the rules were published in 2003 by Curtis Faith).
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
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
