"""Data layer for Study 117 (Pi-Cycle-Top).

Two tapes, one shape (a tz-naive daily OHLCV frame for BTC-USD):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``top_sharpness`` knob
  injects a pronounced local peak (followed by a sharp drawdown) at planted cycle-top
  dates. ``top_sharpness=0`` is a pure random walk — the null hypothesis. With the knob
  dialled up, the synthetic world lets us ask: *even if the indicator worked in theory*,
  would ~3 signal events give us statistical power? (Spoiler: no.)
- ``fetch_daily`` — the real Yahoo! BTC-USD daily tape (``yfinance``), cache-only by
  default so the test-suite never touches the network. Yahoo daily BTC history starts
  2014-09-17; the indicator needs 350 bars of history to warm up, so the effective start
  is late 2015.

No look-ahead: signals are formed on the crossover day *t*; the 'exit' trade enters at
*t+1*'s open.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKER = "BTC-USD"

# The three observable Pi-Cycle-Top signal dates (111-day MA crosses above 2× 350-day MA).
# 2013-11 pre-dates Yahoo history; 2017-06 and 2021-04 are the legendary calls;
# 2024-03 is the most recent.
SIGNAL_DATES = [
    pd.Timestamp("2017-06-21"),  # 2017 cycle top vicinity — Bitcoin ~$2,700
    pd.Timestamp("2021-04-12"),  # 2021 cycle top vicinity — Bitcoin ~$63,000
    pd.Timestamp("2024-03-18"),  # 2024 signal — Bitcoin ~$68,000
]


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 10,
    top_sharpness: float = 0.0,
    top_window_days: int = 60,
    daily_vol: float = 0.04,
    annual_drift: float = 0.60,
    start: str = "2014-01-01",
    seed: int = 117,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with an optional planted cycle-top signal.

    Log returns follow a daily drift-plus-noise process. When ``top_sharpness > 0``,
    three planted "cycle top" events each inject a sharp deceleration: daily drift drops
    to ``-top_sharpness / top_window_days`` for ``top_window_days`` days starting at a
    planted signal date. This is the *only* forecastable structure in the tape:

    - ``top_sharpness = 0``   → a martingale (secular drift only); the Pi-Cycle signal
                                is a fair coin — the null.
    - ``top_sharpness > 0``   → the signal precedes a genuine drawdown.

    The null case ``top_sharpness=0`` is used by the test-suite and for the positive-
    control sweep: when there is nothing to find, the indicator must not find it.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters and the
    signal dates embedded in the tape.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_years * 252), name="date")
    daily_mu = annual_drift / 252.0

    # Build base log-returns
    log_ret = rng.normal(daily_mu, daily_vol, len(idx))

    # Plant synthetic signal events at evenly-spaced thirds of the tape
    n = len(idx)
    planted_signals = [idx[n // 4], idx[n // 2], idx[3 * n // 4]]

    if top_sharpness > 0:
        down_per_day = top_sharpness / top_window_days
        for sig_date in planted_signals:
            for k, dt in enumerate(idx):
                delta = (dt - sig_date).days
                if 0 <= delta < top_window_days:
                    log_ret[k] -= down_per_day

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0.0, daily_vol * 0.4, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(100_000, 5_000_000, close.size).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=idx,
    )

    truth = {
        "top_sharpness": top_sharpness,
        "top_window_days": top_window_days,
        "daily_vol": daily_vol,
        "annual_drift": annual_drift,
        "n_days": len(idx),
        "seed": seed,
        "planted_signals": planted_signals,
        "n_signals": len(planted_signals),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily BTC-USD, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_daily_pc.parquet")


def fetch_daily(
    ticker: str = TICKER,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for BTC-USD; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (the result is cached as a
    parquet under ``_cache/``). Yahoo daily BTC history starts 2014-09-17; the Pi-Cycle
    indicator needs 350 bars of lookback, so the first live signal can fire only in
    late 2015. The three observable signals (2017, 2021, 2024) mean this study lives
    and dies on n≤3 events — the study's structural power ceiling, not a bug.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. "
                f"Call fetch_daily(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy import: only when we go to the network

        raw = yf.download(
            ticker, period="max", interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        # Strip timezone so index is tz-naive throughout
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.sort_index()


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
