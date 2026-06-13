"""Data layer for Study 104 (Bollinger-Reversion).

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A mean-reversion
  strength knob (``reversion``) lets us dial the AR(1)-in-level structure that
  Bollinger-band entries would need to harvest.  At ``reversion = 0`` the log-return
  series is a pure random walk — a fair coin — so the test can assert band entries
  beat random *only* when we plant an effect, and not otherwise.  A second knob
  (``trend_vol_ratio``) controls how "bursty" the price path is, which governs how
  often the bands are pierced.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default
  so the test-suite and the reproducible core never touch the network.  Daily history
  is long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: signals
are formed on closes up to *t*, positions entered at *t+1*'s open.
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
    reversion: float = 0.0,
    annual_vol: float = 0.18,
    start: str = "2010-01-04",
    seed: int = 104,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of mean-reversion.

    Log returns follow a modified AR(1): ``r_t = -reversion * (p_{t-1} - p_mean) + eps_t``
    where ``eps_t`` is i.i.d. normal with daily standard deviation ``annual_vol / sqrt(252)``.
    ``reversion`` is the *only* forecastable structure in the tape:

    - ``reversion = 0``   → a martingale; band-pierce entries are a fair coin.
    - ``reversion > 0``   → mean reversion that the lower-band entry can harvest.
    - ``reversion < 0``   → momentum / trending; band-pierce entries are on the *wrong* side.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = np.empty(n_days)
    # Initialise price at 100; track log-price for the AR(1)-in-log-level specification.
    log_p = np.log(100.0)
    log_p_mean = log_p  # the attractor is the starting level

    for i in range(n_days):
        gap = log_p - log_p_mean      # distance from attractor
        eps = rng.normal(0.0, daily_vol)
        r = -reversion * gap + eps    # revert toward the mean
        log_ret[i] = r
        log_p += r

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
        "reversion": reversion,
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
