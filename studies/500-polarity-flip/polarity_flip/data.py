"""Data layer for Study 500 (Polarity-Flip — old resistance becomes support).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The polarity claim is that once price breaks **above** a prior swing-high resistance, that
  level *flips* to support: the first pullback to it should **bounce up**. We plant exactly
  that: with ``edge > 0`` we keep a rolling level equal to the most-recent confirmed swing high
  that price has already broken above, and whenever the close pulls back *down to* that broken
  level we add a small upward pull (a bounce). With ``edge = 0`` the log-return series is a pure
  random walk and a retest of broken resistance is a fair coin. This is the positive control —
  a harness that cannot bank the planted bounce proves nothing by finding nothing on the real
  tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the resistance
level is a swing high *confirmed* by ``k`` bars on each side (usable only ``k`` bars later),
the break above and the pullback-retest are read on the close of *t*, and the trade is entered
at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs polarity-principle proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pivot_k: int = 10,
    band: float = 0.01,
    start: str = "2010-01-04",
    seed: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of polarity-flip (role-reversal) support.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a polarity-respecting force: we keep a rolling "broken resistance"
    level — the most-recent local swing high that price has already closed *above*. Whenever the
    close pulls back *down into a band* around that broken level (i.e. it retests the old
    resistance from above), we add a small upward pull proportional to ``edge`` — the level
    "holds as support". At ``edge = 0`` the tape is a pure martingale and a retest is a fair
    coin; at ``edge > 0`` a retest of broken resistance is followed by a real bounce that the
    detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    k = pivot_k

    # We simulate a rolling local swing high. We track a short trailing window of recent closes;
    # when a clear local peak forms and price subsequently breaks above it, that peak becomes a
    # "broken resistance" level that should now act as support.
    recent = []                       # trailing log-prices, length up to 2*k+1
    broken_level = None               # current broken-resistance level (log units)
    pending_high = None               # a recent local high not yet broken

    for i in range(n_days):
        recent.append(log_p)
        if len(recent) > 2 * k + 1:
            recent.pop(0)
        # detect a confirmed local high in the trailing window (centre bar is the peak)
        if len(recent) == 2 * k + 1:
            centre = recent[k]
            if centre == max(recent) and centre > recent[k - 1] and centre > recent[k + 1]:
                pending_high = centre
        # a pending high becomes "broken resistance" once price closes above it
        if pending_high is not None and log_p > pending_high + band:
            broken_level = pending_high
            pending_high = None

        pull = 0.0
        if edge > 0.0 and broken_level is not None:
            # retest from above: close pulls back down into a band around the broken level
            if broken_level - band <= log_p <= broken_level + band:
                pull = edge * max(0.0, (broken_level + band) - log_p) + edge * 0.5 * band
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
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
    truth = {"edge": edge, "annual_vol": annual_vol, "pivot_k": pivot_k, "band": band,
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
