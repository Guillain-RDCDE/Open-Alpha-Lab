"""Data layer for Study 188 (Head-Shoulders).

Two tapes, one shape (a timezone-naive daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator.  A ``hs_strength``
  knob lets us embed a controlled head-and-shoulders price path in a background
  random walk, so tests can assert that the detector *fires* when a real pattern is
  planted and *stays quiet* on pure noise.  ``hs_strength=0`` means no pattern
  injected; the tape is a martingale — the null in a bottle.

- ``fetch_daily`` — the real Yahoo! daily OHLCV tape (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network.
  Daily history goes back many years on Yahoo, giving the study genuine statistical
  power.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the
neckline break that *confirms* a pattern is detected at the close of bar *t*; the
forward return begins at bar *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Default basket: large-cap liquid names with long daily histories.
DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "NVDA", "TSLA", "JPM",
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def _inject_hs_top(
    prices: np.ndarray,
    start: int,
    base: float,
    amplitude: float,
    width: int,
) -> None:
    """Overwrite ``prices[start : start + width]`` with a head-and-shoulders top.

    The pattern has five anchor points spread evenly over ``width`` bars:
      - left shoulder at ``base + 0.6 * amplitude`` (bar index ~width//5)
      - neckline dip between left shoulder and head (``base``)
      - head at ``base + amplitude`` (bar index ~2*width//5)
      - neckline dip between head and right shoulder (``base``)
      - right shoulder at ``base + 0.55 * amplitude`` (bar index ~3*width//5)
      - neckline break below ``base`` (last segment, bar index ~4*width//5)

    Values are linearly interpolated between anchors, keeping the injected segment
    smooth and realistic.
    """
    w = width
    anchors = [
        (0,           base),
        (w // 5,      base + 0.60 * amplitude),
        (2 * w // 5,  base),
        (3 * w // 5,  base + 1.00 * amplitude),   # head
        (4 * w // 5,  base),
        (w - 1,       base + 0.55 * amplitude),    # right shoulder
    ]
    xs = np.array([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors])
    t = np.arange(w)
    prices[start : start + w] = np.interp(t, xs, ys)


def synthetic_daily(
    n_days: int = 1000,
    hs_strength: float = 0.0,
    daily_vol_bps: float = 100.0,
    start: str = "2016-01-04",
    n_patterns: int = 3,
    pattern_width: int = 60,
    seed: int = 188,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with optional head-and-shoulders injections.

    Parameters
    ----------
    n_days:
        Total trading days.
    hs_strength:
        When > 0, ``n_patterns`` H&S top patterns are injected at evenly spaced
        locations.  The amplitude of each pattern (peak-to-trough) is
        ``hs_strength * price_at_injection_point``.  ``hs_strength=0`` produces a
        pure random walk — the null hypothesis.
    daily_vol_bps:
        Background log-return standard deviation (basis points per day).
    start:
        First calendar date on the business-day grid.
    n_patterns:
        How many H&S segments to inject (only used when ``hs_strength > 0``).
    pattern_width:
        Number of bars each injected pattern spans.
    seed:
        RNG seed for full reproducibility.

    Returns
    -------
    (bars, truth)
        ``bars`` — daily OHLCV DataFrame; ``truth`` — planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    vol = daily_vol_bps * 1e-4

    # Pure random-walk close series.
    log_ret = rng.normal(0.0, vol, n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    if hs_strength > 0.0 and n_patterns > 0:
        # Space patterns evenly, leaving room at the edges.
        gap = (n_days - n_patterns * pattern_width) // (n_patterns + 1)
        for k in range(n_patterns):
            start_idx = gap * (k + 1) + k * pattern_width
            if start_idx + pattern_width >= n_days:
                break
            base = close[start_idx]
            amplitude = hs_strength * base
            _inject_hs_top(close, start_idx, base, amplitude, pattern_width)

    # Build OHLC from close series.
    open_ = np.empty(n_days)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, vol * 0.5, n_days)) * close
    high = np.maximum(open_, close) + wick
    low_wick = np.abs(rng.normal(0.0, vol * 0.3, n_days)) * close
    low = np.minimum(open_, close) - low_wick
    volume = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "hs_strength": hs_strength,
        "daily_vol_bps": daily_vol_bps,
        "n_days": n_days,
        "n_patterns": n_patterns,
        "pattern_width": pattern_width,
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
    start: str = "2005-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/``).  Daily history goes back to 2005 on
    Yahoo, giving nearly 20 years of bars — far more statistical power than any
    intraday study.

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
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
