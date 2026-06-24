"""Data layer for Study 422 (Elder Ray).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator.  A trend-persistence knob
  (``edge`` = AR(1) coefficient on log returns) plants the only structure an Elder-Ray
  *trend-filter* rule can possibly harvest: positive autocorrelation (momentum).
  ``edge = 0`` is a pure random walk — a fair coin — so a test can assert the Elder-Ray
  timing rule beats buy-and-hold *only* when we plant trend, and reads ~0 when we don't.
  This is the study's null in a bottle (and the positive control).
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first so the
  test-suite and the reproducible core never touch the network.  Daily bars go back
  decades, giving high statistical power compared with intraday studies.

Dr Alexander Elder's *Elder Ray* (1989, *Trading for a Living*) decomposes price around
a 13-period EMA "consensus of value":

    Bull Power(t) = High(t) - EMA13(Close)(t)     (how far bulls push above value)
    Bear Power(t) = Low(t)  - EMA13(Close)(t)     (how far bears push below value)

The folk rule pairs Elder Ray with the EMA *trend*:

- Trend up  (EMA13 rising) AND Bear Power < 0 but *rising*  → go long.
- Trend down (EMA13 falling) AND Bull Power > 0 but *falling* → exit / go short.

We test the long/flat (and, where natural, long/short) timing rule against buy-and-hold,
net of costs, with a one-day execution lag.  No look-ahead is baked in here — that
discipline lives in ``strategy.py`` (signals are formed on closes up to *t*, positions
held over the return of *t+1*).
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
# Synthetic tape — the deterministic offline core / positive control
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 4000,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    drift: float = 0.0003,
    start: str = "2005-01-03",
    seed: int = 422,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable *trend-persistence* knob.

    Log returns follow a bar-level AR(1): ``r_t = drift + edge * (r_{t-1} - drift) + eps_t``
    where ``eps_t`` is i.i.d. normal with standard deviation ``daily_vol``.  A *positive*
    autocorrelation (``edge > 0``) is momentum — the structure an Elder-Ray *trend filter*
    is designed to harvest (stay long while the EMA rises, step aside while it falls).

    - ``edge = 0``   → a martingale with drift; the trend filter is a fair coin and should
                       only collect the drift it happens to be exposed to (≈ buy-and-hold,
                       minus the time spent in cash).
    - ``edge > 0``   → trending tape the long/flat rule can ride (beats buy-and-hold by
                       sidestepping the down-runs).
    - ``edge < 0``   → mean-reverting tape the trend filter is on the *wrong* side of.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = drift + edge * prev + eps[i]
        log_ret[i] = r
        prev = r - drift  # AR(1) on the de-meaned innovation

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar wicks: a small symmetric deviation around the open-close range.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.6, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 80_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    truth = {
        "edge": edge,
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
    return os.path.join(cache_dir, f"elder_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 3,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on miss or ``fetch=True``.

    On a cache miss (or ``fetch=True``) we hit Yahoo via ``yfinance`` with a small backoff,
    then cache the parquet so re-runs are offline.  ``auto_adjust=True`` → total-return
    (dividends reinvested), which we label explicitly downstream.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path) and not fetch:
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw is not None and not raw.empty:
                break
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
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars.dropna()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
