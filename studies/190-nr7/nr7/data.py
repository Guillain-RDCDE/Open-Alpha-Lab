"""Data layer for Study 190 (NR7).

Two tapes, one shape (a daily OHLCV frame):

- ``synthetic_daily`` — a *deterministic, offline* generator. A volatility-clustering
  knob (``vol_persistence``) lets us tune GARCH-like behaviour, so synthetic NR7 days
  cluster when persistence is high and scatter when it is low.  A ``breakout_size``
  knob plants the signal we test: the magnitude by which next-day range expands after
  an NR7.  ``breakout_size=0`` is the null (no expansion); ``breakout_size>0`` plants a
  real effect.  This is the study's null in a bottle.
- ``fetch_daily`` — real Yahoo! daily OHLCV (via yfinance), cache-only by default so
  the test-suite and reproducible core never touch the network.

No look-ahead: signal is formed on closes up to day *t* (the NR7 is identified after
the close of day *t*); next-day range expansion and breakout positions are measured from
day *t+1* onward.
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
    vol_persistence: float = 0.7,
    breakout_size: float = 0.0,
    start: str = "2018-01-02",
    seed: int = 190,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with tunable volatility clustering and NR7 expansion.

    Volatility follows a simple GARCH(1,1)-like recursion:
    ``sigma_t^2 = (1 - vol_persistence) * sigma_bar^2 + vol_persistence * sigma_{t-1}^2``
    so that low-vol days tend to cluster and high-vol days tend to cluster
    (``vol_persistence = 0`` → i.i.d. vol; ``vol_persistence → 1`` → very persistent).

    On NR7 days (the narrowest range of the prior 7 days), the *next* day's range
    is enlarged by a factor ``(1 + breakout_size)`` relative to what it would have
    been otherwise.  ``breakout_size = 0`` is the exact null — no special behaviour
    after an NR7.  ``breakout_size > 0`` plants a genuine volatility-expansion
    effect the engine can find.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_days)

    sigma_bar = 0.012          # ~1.2% daily vol baseline
    sigma = sigma_bar

    opens  = np.empty(n_days)
    highs  = np.empty(n_days)
    lows   = np.empty(n_days)
    closes = np.empty(n_days)
    vols   = rng.integers(500_000, 5_000_000, n_days).astype(float)

    price = 100.0
    nr7_flags = np.zeros(n_days, dtype=bool)  # will be filled after daily ranges known

    # ---- pass 1: draw returns, build OHLC ---------------------------------
    daily_range_mult = np.ones(n_days)  # multiplier applied to range on NR7+1 days
    sigmas = np.empty(n_days)
    for i in range(n_days):
        sigma = np.sqrt(
            (1 - vol_persistence) * sigma_bar**2
            + vol_persistence * sigma**2
        )
        sigmas[i] = sigma

    # Draw returns first; use a two-pass approach to plant NR7 expansion.
    log_rets = rng.normal(0, 1, n_days) * sigmas        # daily close-to-close

    # Intrabar range ~ half-normal scaled to sigma; open is prior close.
    range_noise = np.abs(rng.normal(0, 1, n_days)) * sigmas * 1.5
    upward = rng.random(n_days) > 0.5

    # ---- simulate daily bars sequentially so we can identify NR7 online --
    ranges_stored = np.zeros(n_days)
    price = 100.0
    for i in range(n_days):
        o = price
        r = log_rets[i]
        c = o * np.exp(r)

        # intrabar range expansion on the day AFTER an NR7
        mult = daily_range_mult[i] if i < len(daily_range_mult) else 1.0
        half_r = range_noise[i] * mult
        if upward[i]:
            h = max(o, c) + half_r
            lo = min(o, c)
        else:
            h = max(o, c)
            lo = min(o, c) - half_r

        opens[i]  = o
        highs[i]  = h
        lows[i]   = lo
        closes[i] = c
        ranges_stored[i] = h - lo
        price = c

        # After storing this bar, check if it is NR7 (needs 7 bars of history).
        if i >= 6:
            window_ranges = ranges_stored[i - 6: i + 1]  # includes current
            if window_ranges[-1] == window_ranges.min():
                nr7_flags[i] = True
                # Plant expansion on *next* day
                if i + 1 < n_days and breakout_size > 0:
                    daily_range_mult[i + 1] = 1.0 + breakout_size

    bars = pd.DataFrame(
        {
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": vols,
        },
        index=pd.DatetimeIndex(index, name="date"),
    )
    truth = {
        "n_days":          n_days,
        "vol_persistence": vol_persistence,
        "breakout_size":   breakout_size,
        "seed":            seed,
        "nr7_count":       int(nr7_flags.sum()),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"nr7_{safe}_daily.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    as a parquet under ``_cache/``).  The default ``start`` of 2000-01-01 gives ~25
    years of daily bars — enough for ~3,000 trading days and hundreds of NR7 signals.
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
        bars = bars.dropna(subset=["close"])
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
