"""Data layer for Study 460 (Counterattack / Meeting Lines).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  specific to THIS indicator. A bullish counterattack (meeting) line is two
  opposite-colour candles whose **closes meet** (a down/black candle, then an up/white
  candle that closes back at ~the prior close), appearing after a *downtrend*. The
  believers' claim is that this equal-close "meeting" forecasts a reversal **up**. We
  plant exactly that: with ``edge > 0`` the path gets a real upward kick on the bar
  AFTER any such bullish meeting that follows a down leg; with ``edge = 0`` the
  log-return series is a pure random walk and the meeting is a fair coin. This is the
  positive control — a harness that cannot bank the planted bounce proves nothing by
  finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the downtrend
context and the meeting are read on the close of *t*, and the trade is entered at *t+1*'s
close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Liquid indices/ETFs candlestick proponents draw on: broad tape, big-cap tech, small
# caps, the Dow, and a cross-asset chart. Daily, liquid, long history. Re-used from the
# desk's standard basket so the study runs fully offline against the shared cache.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    trend_lookback: int = 10,
    start: str = "2010-01-04",
    seed: int = 460,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of meeting-line reversal.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    We additionally *engineer* a meeting candle on roughly 1 day in 18: on those bars we set
    the open to gap **down** from the prior close and the close to land back **at the prior
    close** (an equal-close "meeting" between a down candle and an up candle). When that
    meeting follows a genuine down leg (the planted bullish counterattack), with ``edge > 0``
    we add a small upward kick to the NEXT few sessions — the real bounce the detector should
    bank. At ``edge = 0`` the meeting carries no information and an entry is a fair coin.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    log_p = np.log(100.0)
    close[0] = open_[0] = np.exp(log_p)

    # schedule meeting candles (deterministic), and a decaying bounce buffer
    is_meeting = rng.random(n_days) < (1.0 / 18.0)
    bounce_left = 0  # remaining sessions of a planted upward kick
    tol = 0.0015     # how close the meeting close must land to the prior close (log units)

    for i in range(1, n_days):
        prev_close_log = np.log(close[i - 1])
        # is the recent leg DOWN? (planted-context for a *bullish* meeting)
        if i >= trend_lookback:
            down_leg = np.log(close[i - 1]) < np.log(close[i - trend_lookback])
        else:
            down_leg = False

        kick = 0.0
        if bounce_left > 0:
            kick = edge * daily_vol * 3.0   # planted upward drift during the bounce window
            bounce_left -= 1

        eps = rng.normal(0.0, daily_vol)

        if is_meeting[i]:
            # gap the open DOWN, then close back AT the prior close (the meeting)
            gap = abs(rng.normal(0.0, daily_vol * 1.5)) + daily_vol
            open_log = prev_close_log - gap
            close_log = prev_close_log + rng.normal(0.0, tol * 0.4)  # lands ~at prior close
            open_[i] = np.exp(open_log)
            log_p = close_log
            close[i] = np.exp(log_p)
            # arm a bounce only if this bullish meeting followed a down leg
            if edge > 0.0 and down_leg:
                bounce_left = 4
        else:
            open_[i] = close[i - 1]
            log_p = prev_close_log + eps + kick
            close[i] = np.exp(log_p)

    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "trend_lookback": trend_lookback,
             "n_days": n_days, "seed": seed, "tol": tol}
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
