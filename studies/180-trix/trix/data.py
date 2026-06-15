"""Data layer for Study 180 (TRIX).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A bar-level AR(1) knob
  (``momentum``) lets us dial the only thing TRIX can possibly harvest: return persistence
  across multiple smoothing windows.  ``momentum=0`` is a pure random walk — TRIX zero-line
  and signal-line crosses are then fair coins — so a test can assert the strategy beats random
  *only* when we plant a trend, and not otherwise.  This is the study's null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the test-suite and the reproducible core never touch the network.  Daily history goes back
  ~10 years, giving the triple-smoothed indicator enough warmup bars (3 × period) and enough
  independent flips for meaningful inference.

TRIX overview
~~~~~~~~~~~~~
TRIX is the 1-period rate-of-change (percentage change) of a *triple-smoothed* exponential
moving average (EMA) of the close, popularised by Jack Hutson (1983).  With period ``n``:

    EMA1(t) = EMA(close, n)
    EMA2(t) = EMA(EMA1, n)
    EMA3(t) = EMA(EMA2, n)
    TRIX(t) = 100 * (EMA3(t) - EMA3(t-1)) / EMA3(t-1)

The triple smoothing is designed to filter out short-term cycles; only cycles *longer* than
``3n`` bars should survive.  The folk signals are:

- **Zero-line cross** — TRIX crosses from below to above 0 → buy; above to below 0 → sell.
- **Signal-line cross** — TRIX crosses its own EMA(signal) (typically 9-bar) → buy/sell.

The heavy lag is the structural weakness: by the time the triple-smoothed EMA confirms a turn,
much of the move has already happened.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals formed on
closes up to *t*; positions entered at the next bar's open at *t+1*).
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
    n_days: int = 1000,
    momentum: float = 0.0,
    bar_vol_bps: float = 100.0,
    start: str = "2015-01-02",
    seed: int = 180,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable trend-persistence knob.

    Log returns follow a bar-level AR(1): ``r_t = momentum * r_{t-1} + eps_t`` with
    ``eps_t`` i.i.d. normal of standard deviation ``bar_vol_bps`` (daily vol in bps).
    ``momentum`` is the *only* forecastable structure in the tape:

    - ``momentum = 0``   → a martingale; TRIX crosses are fair coins.
    - ``momentum > 0``   → return persistence the trend indicator can ride.
    - ``momentum < 0``   → mean reversion that TRIX (trend-following) is on the *wrong* side of.

    A longer tape (``n_days=1000``) is used than intraday studies because TRIX needs
    ``3 * period`` bars of warmup and we want enough independent flip events for power.

    Bars are stamped on consecutive business days.  Returns ``(bars, truth)`` where
    ``truth`` records the planted parameters for reproducibility and test assertions.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    eps = rng.normal(0.0, bar_vol_bps * 1e-4, n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        r = momentum * prev + eps[i]
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    # Intrabar high/low: a small symmetric wick around the open-close range.
    wick = np.abs(rng.normal(0.0, bar_vol_bps * 0.3e-4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "momentum": momentum,
        "bar_vol_bps": bar_vol_bps,
        "n_days": n_days,
        "n_bars": int(close.size),
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"trix_bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "15y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``).  A 15-year window is requested so that TRIX(15) — which
    needs 45 bars of warmup — has ample independent flip events for meaningful inference.
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
            ticker, period=period, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Ensure tz-naive DatetimeIndex for consistent downstream handling.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
