"""Data layer for Study 453 (Three-Inside-Up / Down).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  specific to THIS indicator. The three-inside-up pattern is: a down day, then an *inside*
  (harami) day whose whole range sits inside the first day's body, then a *confirming* day
  that closes back above the first day's open. The believers' claim is that this triplet
  marks a bullish reversal, so the next bars drift **up**. We plant exactly that: with
  ``edge > 0`` the path receives a small upward push on the bars *following a completed
  three-inside-up*, so the entry harvests a real bounce; with ``edge = 0`` the OHLC series
  is a pure random walk and the pattern's appearance is a fair coin. This is the positive
  control — a harness that cannot bank the planted bounce proves nothing by finding nothing
  on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the pattern is
read on the close of the confirming bar *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The same broad, liquid, long-history tapes the desk reuses so the study runs offline:
# the broad market, big-cap tech, small caps, the Dow, and a cross-asset (gold) chart.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    trend_lookback: int = 5,
    start: str = "2010-01-04",
    seed: int = 453,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of three-inside-up reversal edge.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Open/high/low are synthesised around each close so that genuine *harami* (inside-body)
    days occur. On top of that we plant a pattern-respecting force: whenever the most recent
    three completed bars form a **three-inside-up** (down bar → inside harami → confirming
    close back above bar-1's open, after a short down-trend), we inject a small upward push
    into the *next* few bars proportional to ``edge``. At ``edge = 0`` the tape is a pure
    martingale and the pattern is a fair coin; at ``edge > 0`` a completed three-inside-up is
    followed by a real bounce that the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)

    sessions = pd.bdate_range(start=start, periods=n_days)
    o = np.empty(n_days)
    h = np.empty(n_days)
    lo = np.empty(n_days)
    c = np.empty(n_days)

    log_p = np.log(100.0)
    prev_close = np.exp(log_p)
    push = np.zeros(n_days)   # pending upward push injected after a planted pattern
    o_log = np.empty(n_days)  # log opens, for the harami test below
    c_log = np.empty(n_days)

    for i in range(n_days):
        # open near previous close (small overnight gap)
        gap = rng.normal(0.0, daily_vol * 0.3)
        open_log = np.log(prev_close) + gap
        # intraday drift = random walk step + any planted upward push
        eps = rng.normal(0.0, daily_vol)
        close_log = open_log + eps + push[i]
        oo, cc = np.exp(open_log), np.exp(close_log)
        body_hi, body_lo = max(oo, cc), min(oo, cc)
        wick = np.abs(rng.normal(0.0, daily_vol * 0.4)) * cc
        hi = body_hi + wick
        low = body_lo - wick

        o[i], c[i], h[i], lo[i] = oo, cc, hi, low
        o_log[i], c_log[i] = open_log, close_log
        prev_close = cc
        log_p = close_log

        # ---- plant the edge: detect a completed three-inside-up ending at bar i ----
        if edge > 0.0 and i >= 2 + trend_lookback:
            a, b, d = i - 2, i - 1, i  # first / harami / confirming bars
            # bar a is a down bar in a short downtrend
            downtrend = c[a] < c[a - trend_lookback]
            a_down = c[a] < o[a]
            a_body_hi, a_body_lo = max(o[a], c[a]), min(o[a], c[a])
            # bar b (harami): its whole range sits inside bar a's body
            inside = (h[b] <= a_body_hi) and (lo[b] <= a_body_hi) \
                and (lo[b] >= a_body_lo) and (h[b] >= a_body_lo)
            # bar d (confirmation): closes back above bar a's OPEN (the top of the down body)
            confirm = c[d] > o[a] and c[d] > c[b]
            if downtrend and a_down and inside and confirm:
                # inject a real upward bounce into the next few bars
                horizon = 8
                for j in range(i + 1, min(i + 1 + horizon, n_days)):
                    push[j] += edge * daily_vol

    bars = pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "trend_lookback": trend_lookback,
             "n_days": n_days, "seed": seed}
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
