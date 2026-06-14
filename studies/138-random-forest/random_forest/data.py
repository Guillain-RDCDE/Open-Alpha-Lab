"""Data layer for Study 138 (Random-Forest).

Two tapes, one shape (a daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A tunable ``signal_strength``
  knob embeds a weak predictable component into next-day returns (a scaled lag-1 interaction).
  ``signal_strength = 0`` is a pure geometric random walk — a fair coin — so a walk-forward
  RF can only beat chance when we plant a real signal, and not otherwise.  This is the
  study's null in a bottle.
- ``fetch_daily`` — real Yahoo! daily OHLCV for a configurable set of tickers, cache-first
  by default.  ``fetch=False`` raises if the parquet isn't in ``_cache/``.

No look-ahead: features in ``strategy.py`` use only data available at the close of day *t*;
the position is taken at the open of *t+1*.  The lagged target is formed one step ahead so
the RF never sees the outcome it is predicting during training.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252
DEFAULT_TICKERS = ("SPY",)


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1500,
    signal_strength: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2018-01-02",
    seed: int = 138,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of next-day predictability.

    The generative process::

        r_t = signal_strength * tanh(8 * r_{t-1} * r_{t-2}) + daily_vol * eps_t

    The predictable component is a weak nonlinear function of the two most-recent
    returns (a classic lag interaction a tree can detect).  ``signal_strength = 0``
    collapses to a martingale — the *null*.  ``signal_strength > 0`` plants a
    learnable signal.  Deterministic given ``seed``.

    Returns ``(frame, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")
    eps = rng.standard_normal(n_days)
    r = np.zeros(n_days)
    inv_v2 = 1.0 / (daily_vol * daily_vol)
    for t in range(2, n_days):
        mu = signal_strength * np.tanh(r[t - 1] * r[t - 2] * inv_v2)
        r[t] = mu + daily_vol * eps[t]

    close = 100.0 * np.exp(np.cumsum(r))
    span = np.abs(close) * daily_vol * 0.5
    rng2 = np.random.default_rng(seed + 1)
    high = close + np.abs(rng2.standard_normal(n_days)) * span
    low = close - np.abs(rng2.standard_normal(n_days)) * span
    open_ = np.r_[close[0], close[:-1]]
    volume = 1e7 * np.exp(0.3 * rng2.standard_normal(n_days))

    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    truth = {
        "signal_strength": signal_strength,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace("-", "")
    return os.path.join(cache_dir, f"rf_daily_{safe}.parquet")


def fetch_daily(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``).  With ``fetch=False`` and no cache, raises
    ``FileNotFoundError`` so tests and the offline core never silently get empty data.

    Returns a frame with lowercase columns ``open high low close volume`` indexed by
    ``date`` (timezone-naive).
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
        import yfinance as yf  # lazy: only on explicit fetch

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
        bars.index = pd.to_datetime(bars.index).normalize()
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    bars.index = pd.to_datetime(bars.index)
    bars.index.name = "date"
    return bars.sort_index()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
