"""Data layer for Study 106 (Supertrend).

Two tapes, one shape (a tz-aware OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A bar-level AR(1) drift
  knob (``momentum``) lets us dial the only thing the Supertrend indicator can possibly
  harvest: return persistence.  ``momentum=0`` is a pure random walk — a fair coin —
  so a test can assert the Supertrend flip beats random *only* when we plant a trend,
  and not otherwise.  This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily history is
  long (years of data), giving Supertrend's ATR(10) enough warmup bars and enough
  independent flips for meaningful inference.

Supertrend uses ATR(10, multiplier=3) on the standard settings popularised by
TradingView.  A "flip" is a change of state in the Supertrend direction signal.

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

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic daily tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    momentum: float = 0.0,
    bar_vol_bps: float = 80.0,
    start: str = "2020-01-02",
    seed: int = 106,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of trend persistence.

    Log returns follow a bar-level AR(1): ``r_t = momentum * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal of standard deviation ``bar_vol_bps`` (daily vol in bps).
    ``momentum`` is the *only* forecastable structure in the tape:

    - ``momentum = 0``   → a martingale; the Supertrend flip is a fair coin.
    - ``momentum > 0``   → return persistence the trend indicator can ride.
    - ``momentum < 0``   → mean reversion the Supertrend is on the *wrong* side of.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters for reproducibility / test assertions.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    prev = 0.0
    eps = rng.normal(0.0, bar_vol_bps * 1e-4, n_days)
    for i in range(n_days):
        r = momentum * prev + eps[i]
        log_ret[i] = r
        prev = r

    idx = pd.DatetimeIndex(sessions, name="ts")
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar high/low: symmetric wick around the open→close range.
    wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.3e-4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(500_000, 5_000_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )
    truth = {
        "momentum": momentum,
        "bar_vol_bps": bar_vol_bps,
        "n_days": n_days,
        "n_bars": int(close.size),
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

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  The daily tape is long — ten years of history by
    default — giving Supertrend's ATR(10) enough warmup bars and enough independent
    flips for a meaningful inference.
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
        bars.index.name = "ts"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
