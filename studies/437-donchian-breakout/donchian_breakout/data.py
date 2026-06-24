"""Data layer for Study 437 (Donchian-Breakout).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge
  knob** (``edge``). At ``edge = 0`` the log-return series is a pure random walk —
  a fair coin — so the breakout timing rule must *not* manufacture a Sharpe edge over
  buy-and-hold however the noise falls. At ``edge > 0`` we plant genuine *trend
  persistence* (positive autocorrelation in log-returns) — exactly the structure a
  Donchian breakout is built to harvest — so the harness must light up. This is the
  positive control; it proves the engine *can* bank an edge, and it never backs a
  REAL stamp on real data.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first. Reads the
  cached parquet under ``_cache/`` when present; only an explicit cache miss with
  ``fetch=True`` touches the network (with a small retry/backoff), then caches the
  parquet so re-runs are fully offline. ``auto_adjust=True`` → split/dividend-adjusted
  total-return closes, which matters for a multi-decade buy-and-hold comparison.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the channel
on day *t* uses only data up to *t-1* (``shift``), and the position is entered at *t+1*.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# A small, transparent panel of long-listed, liquid US ETFs. SPY is the headline
# instrument; the rest probe whether the rule generalises across asset classes
# (broad equity, small-cap, tech, bonds, gold, oil, dollar) — the classic
# trend-follower's diversified basket, in spirit.
PANEL = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP"]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core, with a planted-edge knob
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    annual_vol: float = 0.18,
    start: str = "2005-01-03",
    seed: int = 437,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of *trend persistence*.

    Log returns follow an AR(1): ``r_t = edge * r_{t-1} + eps_t`` where ``eps_t`` is
    i.i.d. normal with daily standard deviation ``annual_vol / sqrt(252)``. The
    ``edge`` knob is the *only* forecastable structure in the tape:

    - ``edge = 0``   → a martingale; a Donchian breakout is a fair coin and must NOT
      beat buy-and-hold on a risk-adjusted basis (the null).
    - ``edge > 0``   → positive return autocorrelation (momentum / trend persistence)
      that a breakout entry can harvest — the planted positive control.
    - ``edge < 0``   → mean reversion; the breakout enters on exactly the wrong side.

    Bars are stamped on consecutive business days. Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters — including ``edge``, the knob the test
    asserts against.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    sessions = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, size=n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = edge * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000, 500_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "edge": edge,
        "annual_vol": annual_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first** (offline once cached).

    Reads the cached parquet under ``_cache/`` when present. Network is touched only on
    an explicit cache miss with ``fetch=True`` — then the result is cached as a parquet
    so every subsequent run is offline. A couple of retries with a small backoff guard
    against yfinance rate-limiting. ``auto_adjust=True`` gives split/dividend-adjusted
    total-return closes — essential for a multi-decade buy-and-hold comparison.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
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
    bars.index.name = "date"
    return bars.dropna(subset=["close"])


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
