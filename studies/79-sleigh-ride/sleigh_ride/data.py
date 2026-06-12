"""Data layer for Study 79 (Sleigh-Ride).

Two tapes, one shape (a tz-naive daily OHLCV frame of adjusted closes):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``santa_bps`` knob
  plants an additive drift (in basis points of daily return) on the Santa Claus Rally
  window (last 5 trading days of the year + first 2 of January). ``santa_bps=0`` is a
  fair null; a non-zero value tests that the detection machinery recovers a planted
  effect. This is the study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default
  so the test-suite and the reproducible core never touch the network. For ^GSPC this
  stretches back to 1950, giving ~75 Santa windows — the study's main tape. SPY is a
  shorter cross-check (since 1993).

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the window
is identified by the calendar and always formed *before* the position is opened).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The canonical Hirsch window: last 5 trading days of the year + first 2 of the next.
SANTA_WINDOW_TAIL = 5   # last N trading days of the calendar year
SANTA_WINDOW_HEAD = 2   # first N trading days of the new calendar year


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 75,
    daily_vol_bps: float = 100.0,
    santa_bps: float = 0.0,
    start_year: int = 1950,
    seed: int = 79,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with an optional Santa Claus Rally drift.

    Log returns follow an i.i.d. normal with standard deviation ``daily_vol_bps``
    (in basis points, 1 bp = 1e-4). During every Santa window (last 5 + first 2
    trading days of the year) an extra drift of ``santa_bps * 1e-4`` is added to
    each bar. ``santa_bps = 0`` is a pure random walk — a fair null — so a test can
    assert window detection works only when we plant a signal, and not otherwise.

    Returns ``(bars, truth)`` where ``truth`` records the planted parameters.
    The index covers weekdays only (``pd.bdate_range``) starting 1 January
    ``start_year``; realistic weekend/holiday gaps are approximated well enough for
    the test suite.
    """
    rng = np.random.default_rng(seed)

    # Business days from start_year for n_years full years.
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{start_year + n_years}-01-01")
    dates = pd.bdate_range(start=start, end=end, freq="B")

    n = len(dates)
    # Assign each date to its calendar year.
    years = dates.year

    # Build the Santa window mask: last SANTA_WINDOW_TAIL bdays of year Y
    # + first SANTA_WINDOW_HEAD bdays of year Y+1.
    santa_mask = np.zeros(n, dtype=bool)
    for y in range(start_year, start_year + n_years):
        yr_idx = np.where(years == y)[0]
        if len(yr_idx) >= SANTA_WINDOW_TAIL:
            santa_mask[yr_idx[-SANTA_WINDOW_TAIL:]] = True
        next_yr_idx = np.where(years == y + 1)[0]
        if len(next_yr_idx) >= SANTA_WINDOW_HEAD:
            santa_mask[next_yr_idx[:SANTA_WINDOW_HEAD]] = True

    base_ret = rng.normal(0.0, daily_vol_bps * 1e-4, n)
    # Plant the Santa drift on the window.
    log_ret = base_ret + santa_mask * (santa_bps * 1e-4)

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol_bps * 0.3e-4, n)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(50_000_000, 500_000_000, n).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "n_years": n_years,
        "daily_vol_bps": daily_vol_bps,
        "santa_bps": santa_bps,
        "start_year": start_year,
        "n_bars": int(n),
        "seed": seed,
        "santa_window_bars": int(santa_mask.sum()),
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("=", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = "1950-01-01",
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). The frame is adjusted for splits and dividends (Yahoo
    ``auto_adjust=True``). For ^GSPC, ``start="1950-01-01"`` gives ~75 years of Santa
    windows — the study's main long-horizon tape.
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
        bars.index = pd.DatetimeIndex(bars.index)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
