"""Data layer for Study 73 (First-Light).

Two tapes, one shape (a tz-aware 5-minute OHLCV frame, regular-trading-hours only):

- ``synthetic_5m`` — a *deterministic, offline* generator. An opening-drift knob
  (``open_drift``) lets us plant a directional impulse in the first bar of each session,
  so the ORB rule has something genuine to harvest. ``open_drift=0`` is a fair open with
  no session-start signal, the null in a bottle.
- ``fetch_5m`` — the real Yahoo! 5-minute tape (``yfinance``), cache-only by default so
  the test-suite never touches the network. Yahoo caps sub-hourly history at ~60 calendar
  days; we surface that limitation clearly rather than pretend otherwise.

No look-ahead baked in here — the ORB signal is formed only once the opening-range bar(s)
close; entry is at the *next* bar's open (strategy.py enforces this).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# US-equity RTH 09:30–16:00 ET = 6.5 h = 78 five-minute bars.
BARS_PER_DAY = 78
SESSION_OPEN = "09:30"
SESSION_CLOSE = "15:55"  # last 5-minute bar opens at 15:55


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_5m(
    n_days: int = 60,
    bars_per_day: int = BARS_PER_DAY,
    open_drift: float = 0.0,
    bar_vol_bps: float = 8.0,
    start: str = "2026-01-05",
    seed: int = 73,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible 5-minute OHLCV tape with a tunable opening-range drift.

    Log returns follow a bar-level normal draw (standard deviation ``bar_vol_bps``) with
    a U-shaped intraday vol smile.  The *first bar of each session* receives an additional
    bias of ``open_drift`` (in bps) sampled uniformly ±1 that persists through the first
    N bars — this is the only forecastable structure the ORB rule can harvest:

    - ``open_drift = 0``  → fair open, no session-direction signal; ORB is a coin.
    - ``open_drift > 0``  → the opening tick tips slightly in its own direction; the ORB
      has something real to detect.
    - ``open_drift < 0``  → opening impulse reverses (mean-reversion at open); ORB is on
      the wrong side.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days, tz="America/New_York")

    # Intraday vol smile: fat at open & close, thin midday.
    x = np.linspace(-1.0, 1.0, bars_per_day)
    smile = 0.75 + 0.75 * x ** 2  # ~1.5x at edges, ~0.75x midday

    index = []
    log_ret = np.empty(n_days * bars_per_day)
    k = 0
    for day in sessions:
        open_dt = day.replace(hour=9, minute=30, second=0, microsecond=0)
        stamps = pd.date_range(open_dt, periods=bars_per_day, freq="5min")
        index.extend(stamps)
        eps = rng.normal(0.0, bar_vol_bps * 1e-4, bars_per_day) * smile
        # Opening-range impulse: the first bar gets a drift proportional to open_drift,
        # with direction drawn fresh each day so there is no across-day persistence.
        if open_drift != 0.0:
            direction = rng.choice([-1, 1])
            # Decay the impulse over the first few bars (half-life ~3 bars).
            impulse = open_drift * 1e-4 * direction
            for j in range(bars_per_day):
                decay = np.exp(-j / 3.0)
                log_ret[k] = eps[j] + impulse * decay
                k += 1
        else:
            log_ret[k : k + bars_per_day] = eps
            k += bars_per_day

    idx = pd.DatetimeIndex(index, name="ts")
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar high/low: symmetric wick around open→close.
    wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.5e-4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(5_000, 50_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )
    truth = {
        "open_drift": open_drift,
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

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). Yahoo serves at most ~60 calendar days of 5-minute bars —
    this is the study's structural power ceiling, not a bug. With ``rth_only`` the frame
    is trimmed to the 09:30–16:00 ET regular session.
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
        import yfinance as yf  # lazy: only when going to the network

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
        bars = bars.between_time(SESSION_OPEN, SESSION_CLOSE)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
