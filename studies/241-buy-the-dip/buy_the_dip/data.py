"""Data layer for Study 241 (Buy-the-Dip).

Two tapes, one shape (a tz-naive daily close series):

- ``synthetic_daily`` — a *deterministic, offline* generator. A drift knob and an
  extra mean-reversion knob let us plant precisely the condition that dip-buying
  needs to work: that drawdowns systematically recover. ``reversion=0`` is a pure
  random walk — a fair coin on direction — so a test can confirm that the dip-buying
  rule only outperforms when we deliberately plant recovery patterns.
- ``fetch_daily`` — the real Yahoo! daily close tape (``yfinance``), cache-only by
  default so the test suite never touches the network. SPY from 1993 for maximum
  history.

No look-ahead is baked in here: the rule always checks current drawdown from the
running high and enters at the *next* day's open (strategy.py enforces this).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 2520,
    drift: float = 0.0004,
    daily_vol: float = 0.012,
    reversion: float = 0.0,
    start: str = "2000-01-03",
    seed: int = 241,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable drift and mean-reversion knob.

    Log returns follow: ``r_t = drift + (-reversion * drawdown_t) + eps_t`` where
    ``eps_t`` is i.i.d. N(0, daily_vol). The ``reversion`` term creates a soft
    pull back toward the running high — the structural feature that dip-buying
    needs to exploit.

    - ``reversion = 0``   → pure geometric Brownian motion; dips do NOT systematically
      recover faster than random — dip-buying is a fair coin vs buy-and-hold.
    - ``reversion > 0``   → drawdowns actively pull the price back up; dip-buying has a
      genuine edge (positive carry while sitting in cash is foregone, but entry is
      systematically cheaper).
    - ``drift = 0, reversion = 0`` → symmetric random walk; the strategy with a cash
      allocation that sits out the market loses to full investment.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    log_price = 0.0
    running_high = 0.0
    for i in range(n_days):
        drawdown = min(log_price - running_high, 0.0)  # non-positive
        eps = rng.normal(0.0, daily_vol)
        r = drift + reversion * (-drawdown) + eps  # pull from drawdown
        log_ret[i] = r
        log_price += r
        if log_price > running_high:
            running_high = log_price

    close = 100.0 * np.exp(np.cumsum(log_ret) - log_ret[0])
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "drift": drift,
        "daily_vol": daily_vol,
        "reversion": reversion,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then cached as parquet
    under ``_cache/``). SPY back to 1993 gives over 30 years of history for a
    thorough drawdown analysis.
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
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
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
