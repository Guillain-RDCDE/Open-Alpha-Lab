"""Data layer for Study 436 (Moving-Average Envelopes).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  (``edge`` / ``mean_rev``).  ``edge = 0`` is a pure random walk (a fair coin) so a test
  can assert the envelope timing rule banks an edge *only* when one is planted, and not
  otherwise.  ``edge > 0`` plants AR(1) mean-reversion — the only structure a
  buy-the-lower-band / sell-the-upper-band envelope rule can possibly harvest.  This is the
  study's null (and positive control) in a bottle.
- ``load_real`` / ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-first
  by default so the test-suite and the reproducible core never touch the network.

A **Moving-Average Envelope** (Granville's "moving-average bands", 1960s) plots a simple
moving average and two parallel lines a fixed *percent* above and below it::

    mid(t)   = SMA(close, n)
    upper(t) = mid(t) * (1 + k)        # k = e.g. 0.05 for a 5% envelope
    lower(t) = mid(t) * (1 - k)

Unlike Bollinger Bands (whose half-width is ``m * rolling_std``, i.e. *volatility*-scaled),
the envelope half-width is a *fixed percentage* of the average — it does not breathe with
volatility.  The headline question of this study: **do percent envelopes beat Bollinger
Bands?**  The folk recipes:

- **Mean-reversion** — close pierces the lower band → "oversold", go long; back inside →
  flat.  (The contrarian reading; the one most retail charting sites teach.)
- **Breakout** — close pierces the upper band → trend confirmed, go long; back inside →
  flat.  (The trend-following reading.)

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the band is
computed on closes up to *t*, the position is shifted one bar so it earns the *t+1* return).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (planted-edge knob)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    drift: float = 0.0003,
    start: str = "2010-01-04",
    seed: int = 436,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable mean-reversion (edge) knob.

    The *log price* is built as a slow upward trend plus a mean-reverting deviation — an
    Ornstein–Uhlenbeck / AR(1) process on the **level** (not the one-day return):

        x_t      = (1 - edge) * x_{t-1} + eps_t           (eps i.i.d. N(0, daily_vol))
        log P_t  = drift * t  +  x_t

    ``edge`` is the *reversion speed* of the deviation ``x`` back toward the trend line —
    precisely the multi-day "price strays too far from its average, then snaps back"
    structure a percent envelope (a band around the SMA) is designed to harvest.

    - ``edge = 0``   → a random walk on top of a drift; the envelope timing rule has no
                       persistent band structure to exploit beyond plain beta.
    - ``edge > 0``   → the deviation mean-reverts; buying the lower band catches the snap-back.
    - ``edge < 0``   → an explosive/trending deviation the reversion rule is on the wrong side of.

    Returns ``(data, truth)`` where ``truth["edge"]`` is the ground-truth knob the positive
    control checks (also mirrored as ``truth["mean_rev"]``).
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    x = np.zeros(n_days)
    for i in range(1, n_days):
        x[i] = (1.0 - edge) * x[i - 1] + eps[i]
    trend = drift * np.arange(n_days)
    close = 100.0 * np.exp(trend + x)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    data = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
        "mean_rev": edge,
        "daily_vol": daily_vol,
        "drift": drift,
        "n_days": n_days,
        "seed": seed,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"env_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    On an explicit ``fetch=True`` the result is downloaded with ``auto_adjust=True``
    (total-return adjusted: splits *and* dividends folded into the price) and cached as a
    parquet under ``_cache/``.  A couple of retries with a small backoff guard against
    transient yfinance rate-limits.
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
        import time

        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(3):
            raw = yf.download(
                ticker, period=period, interval="1d", auto_adjust=True, progress=False
            )
            if raw is not None and not raw.empty:
                break
            time.sleep(2.0 * (attempt + 1))
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
    return bars


def load_real(
    tickers=("SPY",),
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Cache-first loader for a small panel: returns ``{ticker: bars}``."""
    return {t: fetch_daily(t, fetch=fetch, cache_dir=cache_dir) for t in tickers}


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
