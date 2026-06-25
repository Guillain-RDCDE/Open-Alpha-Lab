"""Data layer for Study 497 (Woodie's Pivot Points).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  Woodie's pivots are built from yesterday's (H, L, C); the close-weighted pivot
  P=(H+L+2C)/4 spawns a lower support **S1=2P-H**. The believers' claim is that when today's
  price reaches *down to yesterday's S1 it reverts upward*. We plant exactly that: with
  ``edge > 0`` the path is given a real upward kick on any day whose low pierces the prior-day
  S1 (a genuine support bounce the rule can bank); with ``edge = 0`` the log-return series is a
  pure random walk and an S1-touch is a fair coin. This is the positive control — a harness
  that cannot bank the planted bounce proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: Woodie's levels for
day *t* are computed from day *t-1*'s bar (one documented lag), an S1-touch is detected on the
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

# Indices / ETFs Woodie-pivot proponents draw on: the broad tape, big-cap tech, small caps,
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
    seed: int = 497,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of Woodie-S1 support bounce.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a Woodie-S1-respecting force: each day we know yesterday's bar, so
    we can compute yesterday's Woodie pivot P=(H+L+2C)/4 and its lower support S1=2P-H. Whenever
    today's path dips down to (or through) that prior-day S1, we add a small upward pull
    proportional to ``edge`` — a genuine support bounce. At ``edge = 0`` the tape is a pure
    martingale and an S1-touch is a fair coin; at ``edge > 0`` an S1 touch is followed by a real
    bounce that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    open_ = np.empty(n_days)

    log_p = np.log(100.0)
    prev_h = prev_l = prev_c = np.exp(log_p)
    have_prev = False

    for i in range(n_days):
        o = np.exp(log_p)  # open at prior close
        # prior-day Woodie support S1 = 2P - H, P = (H + L + 2C)/4
        pull = 0.0
        s1 = np.nan
        if have_prev:
            P = (prev_h + prev_l + 2.0 * prev_c) / 4.0
            s1 = 2.0 * P - prev_h
        eps = rng.normal(0.0, daily_vol)
        # provisional close before the planted bounce
        log_c = log_p + eps
        c = np.exp(log_c)
        # intraday wick around the open->close path
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * o
        hi = max(o, c) + wick
        lo = min(o, c) - wick
        # planted support: if today's low pierced yesterday's S1, kick the close up
        if edge > 0.0 and have_prev and np.isfinite(s1) and lo <= s1:
            # distance below S1, in log units, pulled back up
            depth = max(0.0, (s1 - lo) / max(s1, 1e-9))
            log_c = log_c + edge * (depth + 0.5 * daily_vol)
            c = np.exp(log_c)
            hi = max(hi, c)
        open_[i] = o
        close[i] = c
        high[i] = hi
        low[i] = lo
        prev_h, prev_l, prev_c = hi, lo, c
        have_prev = True
        log_p = np.log(c)

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
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
