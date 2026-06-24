"""Data layer for Study 432 (Hull Moving Average).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The only structure a moving-average *trend-following* rule can possibly harvest is
  serial trend (positive autocorrelation in the drift).  ``edge = 0`` is a pure random
  walk — a fair coin for a crossover rule — while ``edge > 0`` plants a slow, persistent
  regime-switching drift the HMA can ride.  This is the study's null in a bottle, plus a
  positive control that proves the harness *can* bank a real trend when one is planted.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), cache-first by default so the
  reproducible core and the notebooks never touch the network unless told to.

The Hull Moving Average (Alan Hull, 2005) is a weighted-MA stack designed to be nearly
lag-free::

    WMA(n)  = linearly-weighted moving average over n bars
    HMA(n)  = WMA( 2*WMA(n/2) - WMA(n), period = sqrt(n) )

The folk claim: because HMA hugs price with far less lag than an SMA/EMA, a price-vs-HMA
(or HMA-slope) trend rule fires *earlier* and with *fewer false signals* than a plain SMA.
We turn that into a long/flat (and long/short) daily timing rule and race it, net of costs,
against buy-and-hold and against the obvious simpler benchmark — an SMA crossover.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the position is
formed on closes up to *t* and earns the return of *t+1*: one documented execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core with a planted-edge knob
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 2500,
    edge: float = 0.0,
    daily_vol: float = 0.011,
    regime_len: int = 60,
    start: str = "2014-01-02",
    seed: int = 432,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable *trend* (planted-edge) knob.

    The price is a random walk plus a slow regime-switching drift.  Every ``regime_len``
    bars the drift sign is redrawn; the drift magnitude is ``edge * daily_vol``.  A
    trend-following moving-average rule can only profit from such *persistent* drift —
    so the knob is the exact structure the HMA folklore claims to harvest:

    - ``edge = 0``   → pure martingale; an MA-crossover rule is a fair coin.
    - ``edge > 0``   → persistent trending regimes the HMA can ride (positive control).
    - ``edge < 0``   → choppy/whipsaw tape where a trend rule is on the wrong side.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)

    # Slow regime-switching drift: sign flips every regime_len bars.
    n_regimes = int(np.ceil(n_days / regime_len))
    regime_sign = rng.choice([-1.0, 1.0], size=n_regimes)
    drift = np.repeat(regime_sign, regime_len)[:n_days] * (edge * daily_vol)

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = drift + eps

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
        "edge": edge,
        "daily_vol": daily_vol,
        "regime_len": regime_len,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"hma_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    period: str = "max",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (or an empty cache), then the
    result is cached as a parquet under ``_cache/``.  Yahoo's daily history goes back
    decades, giving ample statistical power.  Bars are total-return adjusted
    (``auto_adjust=True`` folds dividends into the close).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch and os.path.exists(path):
        bars = pd.read_parquet(path)
    else:
        import time

        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(3):
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
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
