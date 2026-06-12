"""Data layer for Study 72 (Loaded-Dice).

Two tapes, one shape (a tz-aware 5-minute OHLCV frame, regular-trading-hours only):

- ``synthetic_5m`` — a *deterministic, offline* generator. A bar-level AR(1) drift
  knob (``momentum``) lets us dial the only thing the SMA-crossover rule can possibly
  harvest: very-short-term return persistence. ``momentum=0`` is a pure random walk —
  a fair coin — so a test can assert the cross beats random *only* when we plant a
  trend, and not otherwise. This is the study's null in a bottle.
- ``fetch_5m`` — the real Yahoo! 5-minute tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network. Yahoo caps
  sub-hourly history at ~60 calendar days, so this is a low-power-by-construction tape;
  we say so loudly rather than pretend otherwise.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*, positions entered at *t+1*'s open).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A US-equity RTH session is 09:30–16:00 ET = 6.5h = 78 five-minute bars.
BARS_PER_DAY = 78
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_5m(
    n_days: int = 60,
    bars_per_day: int = BARS_PER_DAY,
    momentum: float = 0.0,
    bar_vol_bps: float = 8.0,
    start: str = "2026-01-05",
    seed: int = 72,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible 5-minute OHLCV tape with a known amount of micro-trend.

    Log returns follow a bar-level AR(1): ``r_t = momentum * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal of standard deviation ``bar_vol_bps`` (a U-shaped intraday
    vol smile multiplies it, fat at the open and close). ``momentum`` is the *only*
    forecastable structure in the tape:

    - ``momentum = 0``   → a martingale; the SMA cross is a fair coin.
    - ``momentum > 0``   → return persistence the trend rule can ride.
    - ``momentum < 0``   → mean reversion the trend rule is on the *wrong* side of.

    Bars are stamped on consecutive weekday RTH sessions (09:30–16:00 ET). Returns
    ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days, tz="America/New_York")

    # Intraday vol smile: fat at the open & close, thin midday (a gentle U).
    x = np.linspace(-1.0, 1.0, bars_per_day)
    smile = 0.75 + 0.75 * x**2  # ~1.5x at the edges, ~0.75x midday

    index = []
    log_ret = np.empty(n_days * bars_per_day)
    prev = 0.0
    k = 0
    for day in sessions:
        open_dt = day.replace(hour=9, minute=30, second=0, microsecond=0)
        stamps = pd.date_range(open_dt, periods=bars_per_day, freq="5min")
        index.extend(stamps)
        eps = rng.normal(0.0, bar_vol_bps * 1e-4, bars_per_day) * smile
        for j in range(bars_per_day):
            r = momentum * prev + eps[j]
            log_ret[k] = r
            prev = r
            k += 1

    idx = pd.DatetimeIndex(index, name="ts")
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar high/low: a small symmetric wick around the open→close range.
    wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.5e-4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(5_000, 50_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )
    truth = {
        "momentum": momentum,
        "bar_vol_bps": bar_vol_bps,
        "n_days": n_days,
        "bars_per_day": bars_per_day,
        "n_bars": int(close.size),
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo 5-minute, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_5m.parquet")


def fetch_5m(
    ticker: str,
    period: str = "60d",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    rth_only: bool = True,
) -> pd.DataFrame:
    """Real 5-minute OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Yahoo serves at most ~60 calendar days of 5-minute
    bars, so ``period`` above ``"60d"`` is silently clipped by the vendor — this is the
    study's structural power ceiling, not a bug. With ``rth_only`` the frame is trimmed
    to the 09:30–16:00 ET regular session.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached 5m tape for {ticker} at {path}. "
                f"Call fetch_5m({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, period=period, interval="5m", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no 5m bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "ts"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    bars.index = bars.index.tz_convert("America/New_York")
    if rth_only:
        bars = bars.between_time(SESSION_OPEN, "15:55")  # last 5m bar opens 15:55
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
