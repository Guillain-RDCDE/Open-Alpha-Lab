"""Data layer for Study 454 (Three-Outside-Up/Down).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The three-outside pattern is a bullish (bearish) *engulfing* candle confirmed by a third
  candle that closes further in the engulf direction. The believers' claim is that *after a
  confirmed bullish engulf, price keeps rising*. We plant exactly that: with ``edge > 0`` the
  generator detects, on the fly, whenever the last three bars form a bullish three-outside-up
  and adds a small positive drift to the *next* bar's return (and a symmetric negative drift
  after a three-outside-down). With ``edge = 0`` the log-return series is a pure random walk and
  the pattern's entry is a fair coin. This is the positive control — a harness that cannot bank
  the planted continuation proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the pattern is read
on the close of *t* (the confirming bar), and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs candlestick-pattern proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 454,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of three-outside continuation.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Each bar's open is the prior close; wicks are symmetric noise. On top of that we plant a
    three-outside-respecting force: whenever the last three *completed* bars form a bullish
    three-outside-up (a bullish candle engulfing the prior bearish body, then a third bar
    closing still higher), we open a **continuation window**: for the next ``plant_days`` bars we
    add an upward drift of ``edge * daily_vol`` per bar (a three-outside-down opens a symmetric
    downward window). This is exactly the believers' claim — a confirmed engulf forecasts a
    multi-day continuation — so a forward-return entry banks it. At ``edge = 0`` the tape is a
    pure martingale and a three-outside entry is a fair coin; at ``edge > 0`` the pattern is
    followed by a real, bankable continuation.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)
    plant_days = 20   # continuation horizon the planted effect spans

    open_ = np.empty(n_days)
    close = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    log_p = np.log(100.0)
    prev_close = np.exp(log_p)
    cont = 0      # remaining bars of an open up-continuation window (>0) / down (<0)

    for i in range(n_days):
        o = prev_close
        pull = 0.0
        if edge > 0.0 and i >= 3:
            # inspect the three most-recent COMPLETED bars: x=i-3 (engulfed), y=i-2
            # (engulfing), z=i-1 (confirming)
            x_o, x_c = open_[i - 3], close[i - 3]
            y_o, y_c = open_[i - 2], close[i - 2]
            z_c = close[i - 1]
            bull_engulf = (x_c < x_o) and (y_c > y_o) and (y_c >= x_o) and (y_o <= x_c)
            bear_engulf = (x_c > x_o) and (y_c < y_o) and (y_c <= x_o) and (y_o >= x_c)
            if bull_engulf and (z_c > y_c):          # confirmed three-outside-up
                cont = plant_days
            elif bear_engulf and (z_c < y_c):        # confirmed three-outside-down
                cont = -plant_days
            if cont > 0:
                pull = edge * daily_vol
                cont -= 1
            elif cont < 0:
                pull = -edge * daily_vol
                cont += 1
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        c = np.exp(log_p)
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        open_[i] = o
        close[i] = c
        hi[i] = max(o, c) + wick
        lo[i] = min(o, c) - wick
        prev_close = c

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed,
             "plant_days": plant_days}
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
