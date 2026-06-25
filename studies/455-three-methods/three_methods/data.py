"""Data layer for Study 455 (Rising/Falling Three Methods).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The three-methods pattern is a *continuation* signal: a long candle, three small
  counter-candles held inside its range, then a long candle closing past the first — the
  folklore says the original trend *resumes* after the brief consolidation. We plant exactly
  that: with ``edge > 0`` the path is given a small *post-consolidation continuation drift*
  whenever a long candle is followed by a tight three-bar pause inside its range, so a
  trend-direction entry harvests a real continuation; with ``edge = 0`` the log-return series
  is a pure random walk and the pattern's direction is a fair coin. This is the positive
  control — a harness that cannot bank the planted continuation proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the five-candle
pattern is fully *closed* (it ends at bar ``t``), the signal is read on the close of *t*, and
the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs candlestick proponents draw on: the broad tape, big-cap tech, small caps,
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
    seed: int = 455,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of three-methods continuation.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a continuation force keyed to the *exact* three-methods geometry:
    we track a rolling window of the last five candles, and whenever they form a rising (or
    falling) three-methods shape — a long candle, three small candles held inside its range,
    and the close now pushing past the first long candle's close — we add a small drift in the
    breakout direction proportional to ``edge`` over the bars that follow. At ``edge = 0`` the
    tape is a pure martingale and a three-methods entry is a fair coin; at ``edge > 0`` a
    completed pattern is followed by a real continuation that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    log_p = np.log(100.0)
    open_[0] = 100.0
    # continuation drift carried forward for a few bars after a planted pattern fires
    boost = 0.0  # signed log-drift per day, decays
    boost_days = 0

    for i in range(n_days):
        o = np.exp(log_p)
        # planted continuation drift (signed) injected after a completed pattern
        cont = 0.0
        if boost_days > 0:
            cont = boost
            boost_days -= 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + cont
        c = np.exp(log_p)
        open_[i] = o
        close[i] = c
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        hi[i] = max(o, c) + wick
        lo[i] = min(o, c) - wick

        # ---- detect a *just-completed* rising/falling three-methods on bars i-4..i ----
        if edge > 0.0 and i >= 4:
            o0, c0 = open_[i - 4], close[i - 4]
            h0, l0 = hi[i - 4], lo[i - 4]
            body0 = abs(c0 - o0)
            # three middle candles
            mids = range(i - 3, i)
            inside = all(hi[j] <= h0 and lo[j] >= l0 for j in mids)
            small = all(abs(close[j] - open_[j]) < 0.5 * body0 for j in mids) and body0 > 0
            up0 = c0 > o0
            dn0 = c0 < o0
            big_last = abs(c - o) > 0.5 * body0
            # rising: bullish anchor + 3 inside small + last closes above c0
            rising = up0 and inside and small and big_last and c > c0
            falling = dn0 and inside and small and big_last and c < c0
            if rising:
                boost = edge * daily_vol
                boost_days = 4
            elif falling:
                boost = -edge * daily_vol
                boost_days = 4

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
