"""Data layer for Study 477 (Choppiness Index).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The Choppiness Index reads **low** when price travels in a sustained direction (the summed
  per-bar range, sum(ATR), is small relative to the high-low *span* of the window — a straight
  trend), and **high** when price thrashes back and forth (sum of ranges large vs a span that
  barely grows). The believers' claim is that a *low* CI (a "trending" regime) **precedes
  tradable momentum**. We plant exactly that: with ``edge > 0`` the path runs in slow
  directional "trend episodes" during which CI is low *and* a real forward drift continues, so
  a low-CI entry harvests momentum; with ``edge = 0`` the log-return series is a pure random
  walk and a low-CI entry is a fair coin. This is the positive control — a harness that cannot
  bank the planted momentum proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the CI uses only the
trailing window through bar *t*, a low-CI reading is detected on the close of *t*, and the trade
is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs Choppiness-Index proponents apply the regime filter on: the broad tape,
# big-cap tech, small caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    ci_window: int = 14,
    start: str = "2010-01-04",
    seed: int = 477,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of "low-CI precedes momentum".

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that, when ``edge > 0`` we run the tape through slow **trend episodes**: a hidden
    state ``mom`` (a persistent directional bias) switches sign occasionally and, while active,
    adds a steady drift in one direction. During such an episode price moves in a near-straight
    line, so the Choppiness Index reads **low** *and* the forward return keeps drifting the same
    way — exactly the planted structure a low-CI entry should bank. At ``edge = 0`` the bias is
    zero throughout: the tape is a pure martingale and a low-CI reading is a fair coin.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    # The base tape is a pure driftless random walk (so a random-entry baseline earns ~0). The
    # planted structure is the *exact* believers' claim and nothing else: whenever the trailing
    # window is LOW-CI (a clean, straight, recent move), inject a small UPWARD momentum drift for
    # a few bars. So a low-CI reading genuinely *precedes* an up-move that a random day does not
    # see — and the low-CI rule can beat the random baseline. With edge=0 no drift is ever
    # injected and a low-CI reading is a fair coin.
    w = ci_window
    logs = np.empty(n_days)
    boost = 0                               # remaining bars of planted momentum
    for i in range(n_days):
        drift = 0.0
        if edge > 0.0:
            if boost > 0:
                drift = edge * daily_vol * 2.2
                boost -= 1
            elif i >= w:
                # cheap trailing "choppiness" proxy: net move / summed abs moves over last w bars
                seg = logs[i - w:i]
                steps = np.diff(seg, prepend=logs[i - w - 1] if i - w - 1 >= 0 else seg[0])
                gross = np.sum(np.abs(steps)) + 1e-12
                net = abs(seg[-1] - seg[0])
                straightness = net / gross          # ~1 => straight (low CI), ~0 => choppy
                going_up = (seg[-1] - seg[0]) > 0
                if straightness > 0.55 and going_up:
                    boost = 6                        # plant ~6 bars of upward continuation
                    drift = edge * daily_vol * 2.2
                    boost -= 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + drift
        logs[i] = log_p
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
    truth = {"edge": edge, "annual_vol": annual_vol, "ci_window": ci_window,
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
