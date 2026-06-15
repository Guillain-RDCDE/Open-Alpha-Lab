"""Data layer for Study 187 (Three-Soldiers).

Two tapes, one shape (a timezone-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A day-level
  momentum knob (``momentum``) plants a controlled tendency for up-days to
  follow up-days and down-days to follow down-days — the mechanic that *would*
  give Three White Soldiers / Three Black Crows any genuine continuation edge.
  ``momentum = 0`` is a pure random walk; the pattern detector is then a fair
  coin.  This is the study's null in a bottle: a real pattern signal requires
  detectable short-term momentum in the underlying tape.

- ``fetch_daily`` — the real Yahoo! daily OHLCV tape (``yfinance``), cache-only
  by default so the test-suite and the reproducible core never touch the network.
  Daily history is long (many years), giving the study genuine statistical power.

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

# Default basket — broad S&P 500 names to build a cross-section.
DEFAULT_TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    momentum: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2023-01-03",
    seed: int = 187,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of short-term momentum.

    Log returns follow a signed AR(1) model: ``r_t = momentum * sign(r_{t-1}) *
    daily_vol + eps_t`` where ``eps_t`` is i.i.d. normal with std
    ``daily_vol_bps`` (in bps).

    - ``momentum = 0``   → a martingale; Three White Soldiers / Three Black Crows
      read a fair coin.
    - ``momentum > 0``   → controlled momentum; continuation patterns *can* win.
    - ``momentum < 0``   → mean reversion; continuation patterns are on the *wrong*
      side.

    Intrabar OHLC structure is synthesised to produce realistic candlestick bodies:
    consecutive up-bars (forming potential Three White Soldiers) arise naturally when
    positive momentum is planted.  The body fraction of the day's range follows a
    Beta(2, 2) distribution; the open is placed randomly within the prior-close→
    close path to mimic realistic gap behaviour.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Business-day index (Mon–Fri, no holidays), timezone-naive for daily bars.
    dates = pd.bdate_range(start=start, periods=n_days)

    vol = daily_vol_bps * 1e-4
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        # Momentum bias: push the next return in the *same* direction as the prior.
        bias = momentum * np.sign(prev) * vol if prev != 0.0 else 0.0
        r = bias + rng.normal(0.0, vol)
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Synthesise intrabar OHLC — realistic but without planted artifical patterns.
    wick = np.abs(rng.normal(0.0, vol * 0.5, n_days)) * close
    open_frac = rng.uniform(0.05, 0.95, n_days)
    prev_close = np.empty(n_days)
    prev_close[0] = 100.0
    prev_close[1:] = close[:-1]
    open_ = prev_close * np.exp(open_frac * log_ret)
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, vol * 0.3, n_days)) * close
    # Guarantee OHLC validity.
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    vol_shares = rng.integers(500_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_shares},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "momentum": momentum,
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

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``).  Daily history goes back many years
    on Yahoo — this study uses from 2010 onward, giving a long tape for robust
    pattern inference.

    Returns a timezone-naive frame indexed by date with columns
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
        # Drop timezone info for daily bars (dates are calendar dates, not timestamps).
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Normalise index timezone — daily bars are timezone-naive.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
