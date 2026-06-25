"""Data layer for Study 483 (Zero-Lag EMA).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  ZLEMA is a *trend filter*: the rule goes long when price sits above the (de-lagged) moving
  average, betting the trend continues. Its whole selling point is that the de-lag makes it
  react to a fresh trend *sooner* than a plain EMA. So the effect we plant is exactly the thing
  the rule banks: **momentum / trend persistence**. With ``edge = 0`` the log-return series is a
  pure random walk and "price > ZLEMA" is a fair coin (the line is just a lagged average, no
  predictive content). With ``edge > 0`` we add positive return autocorrelation (an AR(1)-style
  trend drift), so once price climbs above its moving average it tends to keep rising — a real
  effect a trend filter can harvest, and one the *faster* (de-lagged) filter should harvest
  sooner. This is the positive control — a harness that cannot bank the planted trend proves
  nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the ZLEMA is causal
(it uses only past closes and a fixed ``lag``), the price-vs-line state is read on the close of
*t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs a ZLEMA trend-follower would run on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    length: int = 20,
    start: str = "2010-01-04",
    seed: int = 483,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of trend persistence (momentum).

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant the effect a trend filter banks: **persistent trend regimes**. A
    slow-moving hidden drift ``mu`` follows its own random walk and is *carried forward* with
    persistence ``rho`` (a long-memory AR(1)), scaled by ``edge`` — so the tape spends stretches
    of many days drifting up or down. Once price has climbed above its moving average during an
    up-regime it tends to keep climbing for the holding horizon. At ``edge = 0`` there is no
    persistent drift and the tape is a pure martingale, so a "price > ZLEMA" entry is a fair
    coin; at ``edge > 0`` being above the (de-lagged) average really does forecast continuation,
    the trend a faster filter should bank sooner.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    rho = 0.99                                   # regime persistence (long memory)
    mu = 0.0                                      # hidden slow drift, a stationary AR(1)
    mu_vol = daily_vol * np.sqrt(1.0 - rho ** 2) * 1.5  # stationary drift sd ~ 1.5*daily_vol
    for i in range(n_days):
        mu = rho * mu + rng.normal(0.0, mu_vol)
        eps = rng.normal(0.0, daily_vol)
        # persistent trend regime: a slow drift `mu` (scaled by `edge`) on top of the noise
        ret = eps + edge * mu
        log_p += ret
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
    truth = {"edge": edge, "annual_vol": annual_vol, "length": length,
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
