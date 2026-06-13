"""Data layer for Study 105 (Coppock-Curve).

Two tapes, one shape (a monthly OHLCV frame, calendar-month closes):

- ``synthetic_monthly`` — a *deterministic, offline* generator. A log-return
  drift knob (``trend_strength``) lets us dial the only thing the Coppock signal
  can possibly harvest: multi-month momentum. ``trend_strength=0`` is a
  random walk — a fair coin — so a test can assert the signal fires correctly
  *only* when we plant a cycle, and not otherwise.
- ``fetch_monthly`` — the real Yahoo! monthly tape (``yfinance``), cache-only by
  default so the test-suite and reproducible core never touch the network. Daily
  bars are fetched and resampled to month-end closes; SPY and ^GSPC are the
  primary tapes.

No look-ahead is baked in here — that discipline lives in ``strategy.py``
(Coppock is computed through month t; the position is taken at month t+1's open,
i.e. the first trading day of month t+1).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Coppock Curve canonical periods (in months)
ROC_PERIOD_LONG = 14   # longer ROC look-back
ROC_PERIOD_SHORT = 11  # shorter ROC look-back
WMA_PERIOD = 10        # weighted moving average of the sum-of-ROC


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 300,
    trend_strength: float = 0.0,
    cycle_period: int = 60,
    monthly_vol: float = 0.04,
    start: str = "2000-01-31",
    seed: int = 105,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly OHLCV tape with a controllable bull/bear cycle.

    Log returns follow a mixture: a baseline random walk plus an optional
    sinusoidal drift of amplitude ``trend_strength`` with ``cycle_period`` months.
    The sine creates the multi-month up/down swings that the Coppock Curve is
    designed to detect.

    - ``trend_strength = 0``  → a martingale; Coppock fires randomly.
    - ``trend_strength > 0``  → cyclic momentum the Coppock signal can find.
    - The null knob is ``trend_strength = 0`` — a fair coin for directional bets.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    Bars have columns ``open, high, low, close, volume`` with a
    ``DatetimeIndex`` at month-end dates.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start, periods=n_months, freq="ME")
    t = np.arange(n_months)

    # Sinusoidal drift representing a long market cycle
    drift = trend_strength * np.sin(2 * np.pi * t / cycle_period)
    noise = rng.normal(0.0, monthly_vol, n_months)
    log_ret = drift + noise

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intramonth high/low: symmetric wick around the open→close range
    wick = np.abs(rng.normal(0.0, monthly_vol * 0.4, n_months)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_months).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "trend_strength": trend_strength,
        "cycle_period": cycle_period,
        "monthly_vol": monthly_vol,
        "n_months": n_months,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily → resample to month-end, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"monthly_{safe}.parquet")


def fetch_monthly(
    ticker: str = "^GSPC",
    start: str = "1950-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real monthly OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Downloads daily bars from Yahoo via ``yfinance`` and resamples them to
    month-end. The daily-bar approach avoids the Yahoo monthly-bar quirks and
    lets us compute the Coppock ROC accurately on genuine month-end closes.
    Network is touched only on an explicit ``fetch=True``; the result is cached
    as a parquet under ``_cache/``.

    The Coppock Curve needs 14 + 10 = 24 months of history before the first
    signal is possible, so we start the fetch as far back as Yahoo allows (1950
    for ^GSPC). ``end=None`` defaults to today.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached monthly tape for {ticker} at {path}. "
                f"Call fetch_monthly({ticker!r}, fetch=True) once to populate the cache."
            )
        return pd.read_parquet(path)

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
    raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    raw.index.name = "date"

    # Resample to month-end: open = first day's open, high = max, low = min,
    # close = last day's close, volume = sum.
    monthly = raw.resample("ME").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    monthly = monthly.dropna(subset=["close"])
    monthly.index.name = "date"

    os.makedirs(cache_dir, exist_ok=True)
    monthly.to_parquet(path)
    return monthly


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
