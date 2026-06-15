"""Data layer for Study 183 (Fisher-Transform).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A bar-level AR(1) knob
  (``mean_rev``) lets us dial the only thing the Fisher-Transform crossover can possibly
  harvest: short-term mean-reversion (or momentum) in the midpoint price.  ``mean_rev=0``
  is a pure random walk — a fair coin — so the test can assert the Fisher/trigger cross
  beats random *only* when we plant a structure, and not otherwise.  This is the study's
  null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily bars go back
  decades, giving ~2,500 bars per instrument for a 10-year window — far more power than
  any intraday study.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*, positions entered at *t+1*'s open).

Ehlers (2002) maps the midpoint of the highest-high / lowest-low range into a Gaussian
via the Fisher transform: ``Fisher = 0.5 * ln((1+x)/(1-x))`` where ``x`` is a clamped
rescaling of the midpoint into (−1, +1) over a rolling look-back.  The transform is
*monotone* in the underlying normalised midpoint, so it cannot add information beyond
the price extremes it is derived from — this is the study's central claim.
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
    n_days: int = 500,
    mean_rev: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2015-01-02",
    seed: int = 183,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable mean-reversion / momentum knob.

    Log returns follow ``r_t = -mean_rev * r_{t-1} + eps_t`` where ``eps_t`` is i.i.d.
    normal with standard deviation ``daily_vol``.  A positive ``mean_rev`` induces
    negative return autocorrelation (mean reversion — the structure the Fisher/trigger
    crossover's buy-oversold / sell-overbought framing tries to harvest).  A negative
    ``mean_rev`` induces momentum (the structure a trend-following framing would harvest).

    - ``mean_rev = 0``  → a martingale; the Fisher signal is a fair coin.
    - ``mean_rev > 0``  → mean-reverting tape; the signal may find a bounce.
    - ``mean_rev < 0``  → trending tape; the signal is on the *wrong* side.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = -mean_rev * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar wicks: a small symmetric deviation around the open-close range.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "mean_rev": mean_rev,
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
    bars per instrument for the 10-year window — far more power than any intraday study.
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
