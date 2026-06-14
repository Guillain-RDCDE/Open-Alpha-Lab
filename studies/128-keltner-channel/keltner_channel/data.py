"""Data layer for Study 128 (Keltner-Channel).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_daily`` — a *deterministic, offline* generator.  Two knobs control the
  forecastable structure:

  - ``momentum``   — bar-to-bar return persistence (AR(1) coefficient on log-returns).
    Positive → trending; negative → mean-reverting; zero → martingale.  The breakout
    arm (enter long when close pierces the upper channel) is only right when momentum > 0.
    The reversion arm (enter long when close pierces the lower channel) is only right when
    momentum < 0.  At exactly zero, *both* rules are fair coins — which is the study's
    null in a bottle.

  - ``annual_vol`` — daily volatility scale (as an annual figure); controls how often the
    channel is pierced and therefore how many trades each arm sees.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily history goes
  back to 2000+ years, giving a meaningful power budget for HAC inference.

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


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    momentum: float = 0.0,
    annual_vol: float = 0.20,
    start: str = "2010-01-04",
    seed: int = 128,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of trend/reversion.

    Log returns follow an AR(1): ``r_t = momentum * r_{t-1} + eps_t`` where
    ``eps_t`` is i.i.d. normal with daily standard deviation ``annual_vol / sqrt(252)``.
    ``momentum`` is the *only* forecastable structure:

    - ``momentum =  0.0`` → a martingale; both Keltner entries are fair coins.
    - ``momentum >  0.0`` → return persistence; breakout arm can harvest it.
    - ``momentum < -0.0`` → mean-reversion; reversion arm can harvest it.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters so tests can confirm symmetry.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        r = momentum * prev + eps
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Intraday wick: a fraction of the daily vol.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000, 500_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "momentum": momentum,
        "annual_vol": annual_vol,
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
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  Daily history goes back to 2000+ for most tickers,
    giving 15–20 years of signal — a meaningful power budget.
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
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Ensure tz-naive index with name "date".
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
