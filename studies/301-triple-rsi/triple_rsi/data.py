"""Data layer for Study 301 (Triple-RSI).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A day-level AR(1) drift
  knob (``reversion``) lets us dial the only thing an RSI(5) oversold-bounce rule can
  possibly harvest: short-term *anti*-persistence (mean reversion). ``reversion=0`` is a
  pure random walk — a fair coin — so a test can assert the rule beats random *only*
  when we plant genuine mean reversion, and not otherwise. This is the study's null in a
  bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py``. The published
recipe says "buy at the close" of the signal bar, so the canonical engine enters at the
signal bar's close; a ``next_open`` variant is offered as a robustness check.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Trading days per year approximation.
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 2000,
    reversion: float = 0.0,
    daily_vol: float = 0.012,
    start: str = "2010-01-04",
    seed: int = 301,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of mean-reversion.

    Log returns follow a day-level AR(1): ``r_t = -reversion * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal of standard deviation ``daily_vol``. Note the **negative**
    sign: ``reversion > 0`` means yesterday's up move predicts a down move today — the
    structure an RSI(5) oversold rule needs to harvest.

    - ``reversion = 0``   → a martingale; the rule is a fair coin.
    - ``reversion > 0``   → mean-reversion the rule can exploit.
    - ``reversion < 0``   → momentum, which works *against* the mean-reversion rule.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        r = -reversion * prev + eps
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    # Synthesise reasonable OHLC from close-to-close moves.
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
        "reversion": reversion,
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
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "1993-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). Daily bars have long history available (decades), which
    is what we need to give the thin Triple-RSI signal enough trades to measure.
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
