"""Data layer for Study 87 (Center-Line).

Two tapes, one shape (a tz-aware 5-minute OHLCV frame, regular-trading-hours only):

- ``synthetic_5m`` — a *deterministic, offline* generator. A mean-reversion knob
  (``reversion``) lets us dial the only thing the VWAP-fade rule can possibly harvest:
  tendency for price to revert toward the session running mean. ``reversion=0`` is a
  pure random walk — the fade is a fair coin. ``reversion > 0`` plants intra-session
  mean-reversion that the VWAP-fade can exploit. This is the study's null in a bottle.
- ``fetch_5m`` — the real Yahoo! 5-minute tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network. Yahoo caps sub-hourly
  history at ~60 calendar days, so this is a low-power-by-construction tape; we say so
  loudly rather than pretend otherwise.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on prices up to bar *t*, positions entered at *t+1*'s open).
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
    reversion: float = 0.0,
    bar_vol_bps: float = 8.0,
    start: str = "2026-01-05",
    seed: int = 87,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible 5-minute OHLCV tape with a known amount of intra-session mean-reversion.

    Log returns follow a modified AR process: each bar's return is pulled toward zero (the
    session mean) by ``reversion``. Specifically each session restarts with price at the prior
    session close, and within the session the deviation from the running mean is damped:
    ``r_t = -reversion * dev_fraction_t + eps_t`` where ``dev_fraction_t`` is the normalised
    distance of price from the session VWAP, scaled by the bar volatility. ``reversion`` is
    the *only* forecastable structure in the tape:

    - ``reversion = 0``   → a martingale; the VWAP-fade is a fair coin.
    - ``reversion > 0``   → intra-session mean-reversion the fade rule can ride.
    - ``reversion < 0``   → price trends away from VWAP (the fade is on the wrong side).

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
    session_start_price = 100.0
    k = 0
    for day in sessions:
        open_dt = day.replace(hour=9, minute=30, second=0, microsecond=0)
        stamps = pd.date_range(open_dt, periods=bars_per_day, freq="5min")
        index.extend(stamps)

        # Each session starts at the previous session close (or 100 on day 1).
        price = session_start_price
        eps_arr = rng.normal(0.0, bar_vol_bps * 1e-4, bars_per_day) * smile

        # Track running cumulative log-return for this session to compute deviation.
        cum_log_ret = 0.0
        for j in range(bars_per_day):
            # Deviation from a flat-session reference (zero drift) scaled to bar vol.
            dev = cum_log_ret / (bar_vol_bps * 1e-4 * np.sqrt(j + 1) + 1e-10)
            r = -reversion * dev * bar_vol_bps * 1e-4 * smile[j] + eps_arr[j]
            log_ret[k] = r
            cum_log_ret += r
            k += 1

        # Session close becomes next session open.
        session_start_price = session_start_price * np.exp(cum_log_ret)

    idx = pd.DatetimeIndex(index, name="ts")
    cum = np.cumsum(log_ret)
    # Recover per-session absolute prices.
    close_arr = np.empty(len(cum))
    session_start_price = 100.0
    k = 0
    for i in range(n_days):
        s = i * bars_per_day
        e = s + bars_per_day
        block = log_ret[s:e]
        cum_block = np.cumsum(block)
        close_arr[s:e] = session_start_price * np.exp(cum_block)
        session_start_price = close_arr[e - 1]

    open_ = np.empty_like(close_arr)
    for i in range(n_days):
        s = i * bars_per_day
        open_[s] = close_arr[s - 1] if s > 0 else 100.0
        open_[s + 1 : s + bars_per_day] = close_arr[s : s + bars_per_day - 1]

    wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.5e-4, close_arr.size)) * close_arr
    hi = np.maximum(open_, close_arr) + wick
    lo = np.minimum(open_, close_arr) - wick
    vol = rng.integers(5_000, 50_000, close_arr.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close_arr, "volume": vol},
        index=idx,
    )
    truth = {
        "reversion": reversion,
        "bar_vol_bps": bar_vol_bps,
        "n_days": n_days,
        "bars_per_day": bars_per_day,
        "n_bars": int(close_arr.size),
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
