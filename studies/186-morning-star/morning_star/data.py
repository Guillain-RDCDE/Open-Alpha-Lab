"""Data layer for Study 186 (Morning-Star).

Two tapes, one shape (a tz-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A day-level
  mean-reversion knob (``reversion``) lets us dial in a controlled tendency for
  down-days to be followed by up-days — the mechanic that *would* give three-candle
  reversal patterns any genuine edge.  ``reversion=0`` is a pure random walk, a fair
  coin for the patterns to read; ``reversion>0`` biases the next bar's direction
  against the prior bar's sign.  The knob also governs how realistic the intrabar
  gap structure looks (morning-star and evening-star require body gaps between the
  first candle and the star, which are produced by scaling the inter-bar gap to the
  daily-vol parameter).

- ``fetch_daily`` — the real Yahoo! daily OHLCV tape (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network.  Daily
  history is long (many years), giving the study genuine statistical power.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (patterns
are identified on the *closing* bar at day *t*; forward returns are measured from
*t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default basket — SPY anchors the study; broad S&P names provide the cross-section.
DEFAULT_TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 500,
    reversion: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2023-01-03",
    seed: int = 186,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a known amount of mean-reversion.

    Log returns follow a sign-level AR(1) model: the next day's return has a
    small tilt *against* the prior day's sign, scaled by ``reversion``.
    Specifically ``r_t = -reversion * sign(r_{t-1}) * |daily_vol| + eps_t``
    where ``eps_t`` is i.i.d. normal with std ``daily_vol_bps`` (in bps).

    - ``reversion = 0``   → a martingale; candlestick patterns read a fair coin.
    - ``reversion > 0``   → controlled mean-reversion; reversal patterns *can* win.
    - ``reversion < 0``   → momentum; reversal patterns are on the *wrong* side.

    Intrabar structure (open/high/low) is synthesised to produce realistic-looking
    candlestick bodies, including inter-day gaps (the morning-star and evening-star
    require a gap between the first candle's body and the star's body).  The open
    of each day is drawn from the prior close plus a small gap noise (±half-vol),
    consistent with real pre-open dislocation.  High/low bracket the open and close
    with a Beta-distributed wick.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Business-day index (Mon–Fri, no holidays), timezone-naïve for daily bars.
    dates = pd.bdate_range(start=start, periods=n_days)

    vol = daily_vol_bps * 1e-4
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        bias = -reversion * np.sign(prev) * vol if prev != 0.0 else 0.0
        r = bias + rng.normal(0.0, vol)
        log_ret[i] = r
        prev = r

    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Open: prior close + a small gap noise (captures overnight gap structure).
    # For the morning-star pattern the gap between candle-1 close and star open is
    # important; drawing the open from a uniform perturbation of the prior close
    # with ±0.5*vol produces realistic intraday-gap distributions without planting
    # a false reversal signal.
    open_ = np.empty(n_days)
    open_[0] = 100.0
    gap_noise = rng.uniform(-vol * 0.5, vol * 0.5, n_days)
    prev_close = np.empty(n_days)
    prev_close[0] = 100.0
    prev_close[1:] = close[:-1]
    open_[1:] = prev_close[1:] * np.exp(gap_noise[1:])

    # High/low: straddle open and close with a small Beta-distributed wick.
    wick_frac = rng.beta(1.5, 3.5, n_days) * vol * 5
    wick = wick_frac * close
    high = np.maximum(open_, close) + wick
    low_wick = rng.beta(1.5, 3.5, n_days) * vol * 5 * close
    low = np.minimum(open_, close) - low_wick
    # Guarantee OHLC validity.
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    vol_shares = rng.integers(500_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol_shares},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "reversion": reversion,
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
        # Drop timezone info for daily bars (dates are calendar dates, not timestamps).
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    # Normalise index timezone — daily bars are timezone-naïve.
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
