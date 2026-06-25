"""Data layer for Study 472 (WaveTrend, LazyBear).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  WaveTrend is an oscillator: it fires a buy when the smoothed channel index (WT1) crosses
  **up** through its signal line (WT2) while *oversold* (deeply negative). The believers'
  claim is that such a cross-up-from-oversold is followed by a real bounce. We plant exactly
  that: with ``edge > 0`` the log path receives a small upward kick on the bars right after
  a fresh oversold cross-up, so the WaveTrend buy harvests a genuine bounce; with ``edge = 0``
  the log-return series is a pure random walk and the cross-up is a fair coin. This is the
  positive control — a harness that cannot bank the planted bounce proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the WaveTrend
lines are causal EMAs/SMAs of past closes, the cross is detected on the close of *t*, and the
trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs WaveTrend proponents draw on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# WaveTrend (LazyBear) defaults, channel/average/signal lengths and the oversold band.
WT_CHANNEL = 10   # n1: EMA length of the typical price (the "channel")
WT_AVERAGE = 21   # n2: EMA length of the smoothed channel index
WT_SIGNAL = 4     # SMA length of WT1 to form the signal line WT2
WT_OVERSOLD = -60.0  # the LazyBear oversold band


# --------------------------------------------------------------------------- #
# WaveTrend on a raw OHLC array — used by the synthetic generator's planted kick
# --------------------------------------------------------------------------- #
def _ema(x: np.ndarray, length: int) -> np.ndarray:
    """Causal EMA with TradingView's alpha = 2/(length+1), seeded at x[0]."""
    a = 2.0 / (length + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _wt1_so_far(tp: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """WT1 (the channel index) computed causally on a typical-price array (no signal line)."""
    esa = _ema(tp, n1)
    d = _ema(np.abs(tp - esa), n1)
    d = np.where(d == 0.0, 1e-12, d)
    ci = (tp - esa) / (0.015 * d)
    return _ema(ci, n2)


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 472,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of WaveTrend-bounce mean reversion.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant a WaveTrend-respecting force: at each step we recompute WT1 on the
    typical price so far, and when WT1 has just crossed **up** through its short SMA signal
    while *oversold*, we add a small positive drift (proportional to ``edge``) over the
    following few bars — a genuine bounce that a WaveTrend buy should bank. At ``edge = 0`` the
    tape is a pure martingale and the cross-up is a fair coin; at ``edge > 0`` a fresh
    oversold cross-up is followed by a real upward move.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    n1, n2, sig, os_band = WT_CHANNEL, WT_AVERAGE, WT_SIGNAL, WT_OVERSOLD

    close = np.empty(n_days)
    tp = np.empty(n_days)             # typical price (here close-only path; hi/lo built after)
    log_p = np.log(100.0)
    boost = np.zeros(n_days)          # remaining planted-bounce drift to inject
    warm = n1 + n2 + sig + 5

    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + boost[i]
        close[i] = np.exp(log_p)
        tp[i] = close[i]
        # Recompute WT1 on the path so far and look for a fresh oversold cross-up at bar i.
        if edge > 0.0 and i >= warm:
            wt1 = _wt1_so_far(tp[: i + 1], n1, n2)
            if wt1.size > sig + 1:
                wt2 = np.convolve(wt1, np.ones(sig) / sig, mode="valid")
                # align: wt2[k] corresponds to wt1 index k + sig - 1
                j = wt1.size - 1                       # current bar in wt1 space
                jj = j - (sig - 1)                     # current bar in wt2 space
                if jj >= 1:
                    crossed_up = (wt1[j] > wt2[jj]) and (wt1[j - 1] <= wt2[jj - 1])
                    oversold = wt1[j - 1] < os_band
                    if crossed_up and oversold:
                        # plant a bounce over the next ~5 bars
                        for h in range(1, 6):
                            if i + h < n_days:
                                boost[i + h] += edge * daily_vol * (6 - h)

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
    truth = {"edge": edge, "annual_vol": annual_vol, "n1": n1, "n2": n2,
             "signal": sig, "oversold": os_band, "n_days": n_days, "seed": seed}
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
