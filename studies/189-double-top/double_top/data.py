"""Data layer for Study 189 (Double-Top / Double-Bottom).

Two tapes, one shape (a timezone-naïve daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A scalar
  ``reversal`` knob plants a known amount of mean-reversion into the log-return
  process: ``reversal = 0`` is a pure random walk (the double-top pattern reads
  noise); ``reversal > 0`` biases the *next* bar against the prior bar's sign,
  creating the kind of short-term reversion a confirmed reversal pattern could
  exploit.  This is the study's null in a bottle — if the detector cannot find
  an edge even on a tape with planted reversion, the engine is broken; if it
  finds one only on the synthetic tape and not on the real tape, the edge is not
  there.

- ``fetch_daily`` — the real Yahoo! daily OHLCV tape (``yfinance``), cache-only
  by default so the test-suite and the reproducible core never touch the network.
  Daily history is long (15+ years), giving genuine statistical power.

No look-ahead is baked in here — that discipline lives in ``strategy.py``
(patterns are identified on the *closing* bar at day *t*; forward returns are
measured from *t+1* onward).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default basket — broad cross-section of US equities / indices.
DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL",
    "JPM", "XOM", "GLD", "IWM",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 1000,
    reversal: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2018-01-02",
    seed: int = 189,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable amount of mean-reversion.

    Log returns follow a sign-biased AR(1):
    ``r_t = −reversal · sign(r_{t-1}) · daily_vol + eps_t``
    where ``eps_t`` is i.i.d. normal with standard deviation ``daily_vol_bps``
    (in bps).

    - ``reversal = 0``   → a martingale; double-top/bottom patterns read noise.
    - ``reversal > 0``   → controlled mean-reversion; reversal patterns *can* win.
    - ``reversal < 0``   → momentum; reversal patterns are on the *wrong* side.

    Intrabar structure (open/high/low) is synthesised to produce realistic
    candlestick shapes without planting spurious signals.  Returns
    ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    vol = daily_vol_bps * 1e-4
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        bias = -reversal * np.sign(prev) * vol if prev != 0.0 else 0.0
        r = bias + rng.normal(0.0, vol)
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Synthesise intrabar OHLC — realistic but without planted double-top signals.
    prev_close = np.empty(n_days)
    prev_close[0] = 100.0
    prev_close[1:] = close[:-1]
    open_frac = rng.uniform(0.05, 0.95, n_days)
    open_ = prev_close * np.exp(open_frac * log_ret)
    wick_up = np.abs(rng.normal(0.0, vol * 0.4, n_days)) * close
    wick_dn = np.abs(rng.normal(0.0, vol * 0.4, n_days)) * close
    high = np.maximum(open_, close) + wick_up
    low = np.minimum(open_, close) - wick_dn
    # Guarantee valid OHLC ordering.
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.integers(500_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "reversal": reversal,
        "daily_vol_bps": daily_vol_bps,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily OHLCV, cache-only by default
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

    Network is touched only on an explicit ``fetch=True`` call (then the result
    is cached as a parquet under ``_cache/``).  Daily history goes back many
    years on Yahoo — this study uses 2010-onward, giving 15 years of daily bars
    for robust inference.

    Returns a timezone-naïve frame indexed by date with columns
    ``open, high, low, close, volume`` (all lowercase).
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
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Normalise — daily bars are timezone-naïve.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
