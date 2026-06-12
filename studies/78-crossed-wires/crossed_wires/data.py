"""Data layer for Study 78 (Crossed-Wires).

Two tapes, one shape (a tz-aware daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A bar-level AR(1) drift
  knob (``momentum``) lets us dial the only thing MACD can possibly harvest: medium-term
  return persistence across the 12/26 fast/slow EMA horizons. ``momentum=0`` is a pure
  random walk — a fair coin — so a test can assert the MACD signal beats random *only*
  when we plant a trend, and not otherwise.  This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily history is
  long (years of data), giving MACD's sluggish indicators enough warmup and enough
  out-of-sample bars to distinguish noise from edge.

Two additional helpers mirror Study 72 for the 5-minute tape so the cross-study
comparison ("same indicator at two horizons") is apples-to-apples:

- ``synthetic_5m`` — identical contract to Study 72's generator but seeded differently
  (seed=78) so the two studies are independent.
- ``fetch_5m`` — the real Yahoo! 5-minute tape, same ~60-day cap as Study 72.

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

# Daily tape constants
TRADING_DAYS_PER_YEAR = 252

# 5-minute tape constants — mirrors Study 72
BARS_PER_DAY = 78
SESSION_OPEN = "09:30"
SESSION_CLOSE = "16:00"


# ---------------------------------------------------------------------------
# Synthetic daily tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    momentum: float = 0.0,
    bar_vol_bps: float = 80.0,
    start: str = "2020-01-02",
    seed: int = 78,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of trend persistence.

    Log returns follow a bar-level AR(1): ``r_t = momentum * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal of standard deviation ``bar_vol_bps`` (daily vol in bps).
    ``momentum`` is the *only* forecastable structure in the tape:

    - ``momentum = 0``   → a martingale; the MACD cross is a fair coin.
    - ``momentum > 0``   → return persistence the trend-following MACD can ride.
    - ``momentum < 0``   → mean reversion the MACD is on the *wrong* side of.

    Bars are stamped on consecutive business days. Returns ``(bars, truth)`` where
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
    # Intrabar high/low: symmetric wick around open→close range.
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
# Synthetic 5-minute tape — mirrors Study 72 for cross-study comparison
# ---------------------------------------------------------------------------
def synthetic_5m(
    n_days: int = 60,
    bars_per_day: int = BARS_PER_DAY,
    momentum: float = 0.0,
    bar_vol_bps: float = 8.0,
    start: str = "2026-01-05",
    seed: int = 78,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible 5-minute OHLCV tape — same contract as Study 72 but seed=78.

    Uses a bar-level AR(1) with ``momentum`` controlling return persistence, identical
    to Study 72's ``synthetic_5m`` so both studies can be compared apples-to-apples.
    Bars are stamped on consecutive weekday RTH sessions (09:30–16:00 ET).
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days, tz="America/New_York")

    x = np.linspace(-1.0, 1.0, bars_per_day)
    smile = 0.75 + 0.75 * x**2  # U-shaped intraday vol smile

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
# Real tapes — Yahoo daily and 5-minute, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path_daily(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def _cache_path_5m(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_5m.parquet")


def fetch_daily(
    ticker: str,
    period: str = "5y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  Unlike the 5-minute tape, the daily tape is long —
    typically five or more years of history — giving MACD's 26-bar EMA enough warmup
    bars and enough independent trades for a meaningful inference.
    """
    path = _cache_path_daily(ticker, cache_dir)
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


def fetch_5m(
    ticker: str,
    period: str = "60d",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    rth_only: bool = True,
) -> pd.DataFrame:
    """Real 5-minute OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Yahoo serves at most ~60 calendar days of 5-minute bars — a structural power
    ceiling, not a bug.  Used for the cross-study comparison with Study 72 (same
    instruments, same resolution, different indicator).
    """
    path = _cache_path_5m(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached 5m tape for {ticker} at {path}. "
                f"Call fetch_5m({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf

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
        bars = bars.between_time(SESSION_OPEN, "15:55")
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
