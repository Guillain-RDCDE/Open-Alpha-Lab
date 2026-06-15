"""Data layer for Study 176 (Hot-Hand).

Two tapes, one shape (a daily OHLCV-like frame of S&P 500 index prices):

- ``synthetic_daily`` — a *deterministic, offline* generator. An AR(1) return
  persistence knob (``autocorr``) lets us dial the only thing streak-following
  can harvest: return serial dependence. ``autocorr=0`` is a fair coin flip each
  day — the gambler's null. ``autocorr>0`` means the hot hand is real (momentum);
  ``autocorr<0`` means the gambler's fallacy is right (mean reversion). This is
  the study's null in a bottle.
- ``load_gspc`` — the real S&P 500 daily tape from Yahoo! Finance / yfinance,
  cache-only by default so the test-suite and the reproducible core never touch
  the network.

No look-ahead: streaks are identified on close-to-close returns up to day *t*;
the bet is on day *t+1*.
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
    n_days: int = 1000,
    autocorr: float = 0.0,
    annual_vol: float = 0.16,
    annual_drift: float = 0.07,
    start: str = "2000-01-03",
    seed: int = 176,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily return series with a known amount of serial dependence.

    Log returns follow a daily AR(1): ``r_t = autocorr * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal, scaled so the unconditional vol is ``annual_vol``.
    ``autocorr`` is the *only* forecastable structure in the tape:

    - ``autocorr = 0``   → a martingale; streaks carry no information.
    - ``autocorr > 0``   → return persistence (hot-hand is real; momentum pays).
    - ``autocorr < 0``   → mean reversion (gambler's fallacy is right; contrarian pays).

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # business days only
    dates = pd.bdate_range(start=start, periods=n_days)

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252
    # For AR(1) with autocorr rho, innovations have std = daily_vol * sqrt(1 - rho^2)
    innov_std = daily_vol * np.sqrt(max(1.0 - autocorr ** 2, 1e-10))

    log_rets = np.empty(n_days)
    prev = 0.0
    eps = rng.normal(0.0, innov_std, n_days)
    for t in range(n_days):
        r = daily_drift + autocorr * prev + eps[t]
        log_rets[t] = r
        prev = r - daily_drift  # centre the AR on zero

    close = 100.0 * np.exp(np.cumsum(log_rets))
    bars = pd.DataFrame({"close": close}, index=dates)
    bars.index.name = "date"
    truth = {
        "autocorr": autocorr,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily OHLCV, cache-only by default
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("=", "").replace("/", "")
    return os.path.join(cache_dir, f"daily_{safe}.parquet")


def load_gspc(
    ticker: str = "^GSPC",
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real S&P 500 daily OHLCV; cache-only unless ``fetch=True``.

    Yahoo provides daily price history from ~1950 for ^GSPC. The default
    ``start="1993-01-01"`` covers three decades of data — long enough to
    count streaks of length 6 without running out of sample. Network is
    touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).

    Returns a DataFrame with a ``close`` column and a timezone-naive Date index.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_gspc(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["close"]]
        if hasattr(bars.index, "tz") and bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if hasattr(bars.index, "tz") and bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
