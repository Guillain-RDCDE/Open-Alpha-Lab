"""Data layer for Study 426 — Know Sure Thing (KST).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-trend knob**
  (``trend``).  KST is a long-term momentum oscillator, so the only structure its
  crossover rule can possibly harvest is **persistent trend / positive return
  autocorrelation**.  ``trend = 0`` is a pure random walk — a fair coin — so a test can
  assert the KST timing rule beats buy-and-hold *only* when we plant trend, and not
  otherwise.  This is the study's null in a bottle.  Returns ``(data, truth_dict)``.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads the
  cached parquet under ``_cache/`` and only hits the network on an explicit cache miss
  (with a small retry/backoff), then caches the result so re-runs are offline.

KST (Pring, 1992) is defined on daily closes as::

    ROC_i      = close / close.shift(roc_i) - 1            (i = 1..4)
    smoothed_i = SMA(ROC_i, sma_i)
    KST        = 1*smoothed_1 + 2*smoothed_2 + 3*smoothed_3 + 4*smoothed_4
    signal     = SMA(KST, sig_n)

with Pring's daily parameters ROC = (10, 15, 20, 30), SMA = (10, 10, 10, 15), signal = 9.
The folk rule: BUY when KST crosses *above* its signal line, SELL/flat when it crosses
below.  No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals
are formed on closes up to *t*, the position is held for the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (with the planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    trend: float = 0.0,
    daily_vol: float = 0.011,
    start: str = "2005-01-03",
    seed: int = 426,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable trend (momentum) knob.

    Log returns follow a bar-level AR(1): ``r_t = trend * r_{t-1} + eps_t`` where ``eps_t``
    is i.i.d. normal with standard deviation ``daily_vol`` and a small positive drift.  A
    *positive* return autocorrelation (positive ``trend``) is the structure that KST's
    momentum-crossover rule is designed to harvest — runs in one direction persist, so a
    trend-follower can ride them.

    - ``trend = 0``   → a martingale (with drift); the KST timing rule is a fair coin and
                        cannot beat buy-and-hold after costs.
    - ``trend > 0``   → trending tape the KST crossover can ride (the positive control).
    - ``trend < 0``   → mean-reverting tape the trend rule is on the *wrong* side of.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    drift = 0.0003  # ~7.5%/yr equity-like drift so buy-and-hold is a fair benchmark
    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = drift + trend * prev + eps[i]
        log_ret[i] = r
        prev = r - drift  # autocorrelate the *shock*, not the drift

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
        "trend": trend,
        "daily_vol": daily_vol,
        "drift": drift,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"kst_{safe}_1d.parquet")


def have_real(ticker: str = "SPY", cache_dir: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(_cache_path(ticker, cache_dir))


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool | None = None,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; **cache-first**.

    Reads the cached parquet under ``_cache/`` if present.  On a cache miss (or with an
    explicit ``fetch=True``) it downloads via yfinance with a small retry/backoff and
    caches the parquet so re-runs are offline.  Auto-adjusted closes (total return).

    SPY launched 1993-01-29 — the longest clean US-equity ETF daily tape on Yahoo.
    """
    path = _cache_path(ticker, cache_dir)
    if fetch is None:
        fetch = not os.path.exists(path)

    if not fetch:
        bars = pd.read_parquet(path)
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
            time.sleep(1.5 * (attempt + 1))  # small backoff
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
    return bars.dropna()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
