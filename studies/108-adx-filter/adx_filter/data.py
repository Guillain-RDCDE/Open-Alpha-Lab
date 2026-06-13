"""Data layer for Study 108 (ADX-Filter).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A day-level AR(1) drift
  knob (``trend_strength``) controls the degree of return persistence.  When
  ``trend_strength = 0`` the series is a martingale; when it is positive the series
  trends, and a directional rule (MA cross or Donchian) gains signal.  A separate
  ``adx_efficiency`` knob (default 1.0) scales how faithfully ADX tracks the planted
  trend: at 1.0 the ADX naturally rises in trending regimes; at 0.0 it is pure noise
  uncorrelated with actual trend strength — the study's null for "ADX adds nothing."
  This is the null hypothesis in a bottle for the meta-question: does ADX gate *improve*
  a trend rule beyond what the rule itself already delivers?

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and reproducible core never touch the network.  Daily history is long
  (many years) and supports a more powerful test than the 5-minute cap in Study 72.

No look-ahead is baked in here — signals are formed on closes up to day *t*, positions
entered at day *t+1*'s open.
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
    trend_strength: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2018-01-02",
    seed: int = 108,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known degree of trend persistence.

    Log returns follow a day-level AR(1):
    ``r_t = trend_strength * r_{t-1} + eps_t`` with ``eps_t`` i.i.d. normal of
    standard deviation ``daily_vol``.  ``trend_strength`` is the *only* forecastable
    structure in the tape:

    - ``trend_strength = 0``   → a martingale; any trend rule is a fair coin.
    - ``trend_strength > 0``   → return persistence that a MA cross or Donchian can harvest.
    - ``trend_strength < 0``   → mean reversion; the trend rule is on the *wrong* side.

    Bars are stamped on consecutive weekday business days starting at ``start``.
    OHLCV is synthesised from the log-return path: open = prior close, high/low add
    symmetric intrabar wicks, volume is random.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    prev = 0.0
    eps = rng.normal(0.0, daily_vol, n_days)
    for i in range(n_days):
        r = trend_strength * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intraday wicks: a small symmetric addition around the open→close range.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.4, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(index, name="date"),
    )
    truth = {
        "trend_strength": trend_strength,
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
    start: str = "2010-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    as a parquet under ``_cache/``).  With daily bars the full history back to 2010
    is available, giving a high-powered test compared to the 60-day cap on 5-minute
    bars.
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
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
