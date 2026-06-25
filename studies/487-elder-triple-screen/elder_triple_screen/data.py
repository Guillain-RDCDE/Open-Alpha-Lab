"""Data layer for Study 487 (Elder's Triple Screen).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The triple-screen long fires when (1) the slow weekly trend is up, (2) a fast daily
  oscillator is oversold (a pullback against the trend), and (3) price breaks out above the
  prior bar's high. The believers' claim is that *this exact pullback-in-an-uptrend, bought on
  the breakout, bounces*. We plant exactly that: with ``edge > 0`` the path gets a real upward
  kick on the bars immediately *after* a genuine triple-screen alignment, so the rule harvests
  a true bounce; with ``edge = 0`` the log-return series is a pure random walk and the entry is
  a fair coin. This is the positive control — a harness that cannot bank the planted bounce
  proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the weekly trend is
read on completed weekly bars (shifted one day), the oscillator and breakout are read on the
close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs triple-screen proponents trade: the broad tape, big-cap tech, small caps,
# and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 487,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of triple-screen bounce.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``,
    overlaid on a slow rising-then-falling regime so the weekly-trend screen has something to
    bite on. On top of that we plant a triple-screen-respecting force: we track a slow trend
    proxy and a fast oscillator proxy, and whenever the path is in a genuine *pullback within an
    up-trend* (slow up, fast oversold) we add a small upward pull on the next bar proportional
    to ``edge``. At ``edge = 0`` the tape is a pure martingale and the triple-screen entry is a
    fair coin; at ``edge > 0`` a valid alignment is followed by a real bounce the detector
    should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    # Slow regime: a gentle, mean-reverting trend component so weeks have an actual direction.
    trend = 0.0
    trend_target = rng.normal(0.0, daily_vol * 0.6, n_days)  # where the slow trend wants to go
    fast = 0.0          # an EWMA of recent returns standing in for the daily oscillator
    prev_close = 100.0  # prior bar's close (proxy for the prior high, drives the breakout)
    was_oversold = 0    # bars since the last oversold (Screen-2) reading
    boost = 0           # remaining days of planted post-entry bounce

    for i in range(n_days):
        # slow trend wanders toward its target (regime)
        trend += 0.02 * (trend_target[i] - trend)
        cur = np.exp(log_p)
        # detect a genuine triple-screen-like alignment as we go (no future data):
        #   Screen 1: trend up; Screen 2: oversold within last few bars; Screen 3: breakout.
        if fast < -0.5 * daily_vol:
            was_oversold = 5
        breakout = cur > prev_close
        if edge > 0.0 and trend > 0.0 and was_oversold > 0 and breakout:
            boost = 6  # plant a multi-day bounce starting now (captured by forward returns)
        pull = edge * daily_vol * 2.0 if boost > 0 else 0.0
        eps = rng.normal(0.0, daily_vol)
        step = trend + eps + pull
        log_p += step
        fast = 0.6 * fast + 0.4 * step  # update fast oscillator proxy
        prev_close = cur
        was_oversold = max(0, was_oversold - 1)
        boost = max(0, boost - 1)
        close[i] = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed}
    return bars, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}) once (network) to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close"]]


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
