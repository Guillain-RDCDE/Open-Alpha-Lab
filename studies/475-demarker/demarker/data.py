"""Data layer for Study 475 (DeMarker).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  DeMark's DeMarker oscillator falls toward 0 when recent **highs stop making new highs**
  (the up-moves dry up) and rises when the lows stop making new lows. A reading below 0.3 is
  the believers' "oversold exhaustion": price is supposed to bounce up. We plant exactly that:
  with ``edge > 0`` the path is pulled **upward** in proportion to how *low* the rolling
  DeMarker has fallen (i.e. after a run of weak highs/falling lows), so a "DeMarker rising out
  of oversold" entry harvests a real rebound; with ``edge = 0`` the log-return series is a pure
  random walk and the entry is a fair coin. This is the positive control — a harness that cannot
  bank the planted rebound proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the DeMarker at bar
*t* uses only data through *t*, the "rising out of oversold" trigger is read on the close of
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

# Indices / ETFs DeMarker proponents draw on: the broad tape, big-cap tech, small caps, and a
# couple of cross-asset charts. Daily, liquid, long history. Reused from the desk's idiom so the
# study runs fully offline from the shared cache.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

DEMARKER_PERIOD = 14  # DeMark's classic look-back


# --------------------------------------------------------------------------- #
# A tiny offline DeMarker (mirrors strategy.demarker) used only to plant the edge
# --------------------------------------------------------------------------- #
def _rolling_demarker(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """DeMarker over arrays, returns an array aligned to ``high`` (NaN until warmed up)."""
    n = high.size
    de_max = np.zeros(n)
    de_min = np.zeros(n)
    de_max[1:] = np.maximum(high[1:] - high[:-1], 0.0)
    de_min[1:] = np.maximum(low[:-1] - low[1:], 0.0)
    out = np.full(n, np.nan)
    for i in range(period, n):
        sm_max = de_max[i - period + 1:i + 1].sum()
        sm_min = de_min[i - period + 1:i + 1].sum()
        denom = sm_max + sm_min
        out[i] = (sm_max / denom) if denom > 0 else 0.5
    return out


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    period: int = DEMARKER_PERIOD,
    bounce_days: int = 20,
    start: str = "2010-01-04",
    seed: int = 475,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of DeMarker-exhaustion mean reversion.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that we plant *exactly the effect the believers claim*: whenever the **rolling
    DeMarker turns up out of oversold** (it was below 0.3 yesterday and is higher today — the very
    trigger :func:`strategy.oversold_rising_entries` reads), we add a small **upward** drift
    spread over the next ``bounce_days`` bars, proportional to ``edge``. The kick is keyed to the
    *trigger event*, not held during oversold, so it does not suppress the oversold reading
    itself — it lands in the forward window the detector actually measures. At ``edge = 0`` the
    tape is a pure martingale and a DeMarker entry is a fair coin; at ``edge > 0`` an
    oversold-rising turn is followed by a real rebound the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    log_p = np.log(100.0)
    close[0] = np.exp(log_p)
    high[0] = close[0]
    low[0] = close[0]

    de_max_buf = np.zeros(n_days)
    de_min_buf = np.zeros(n_days)
    bounce = np.zeros(n_days)            # scheduled upward drift, keyed to past triggers
    dem_prev2 = 0.5                       # DeMarker at i-2
    dem_prev1 = 0.5                       # DeMarker at i-1
    per_day = (edge * daily_vol * 1.5) / max(bounce_days, 1)

    for i in range(1, n_days):
        log_p += rng.normal(0.0, daily_vol) + bounce[i]
        close[i] = np.exp(log_p)
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * close[i]
        prev = close[i - 1]
        high[i] = max(prev, close[i]) + wick
        low[i] = min(prev, close[i]) - wick
        de_max_buf[i] = max(high[i] - high[i - 1], 0.0)
        de_min_buf[i] = max(low[i - 1] - low[i], 0.0)

        # rolling DeMarker known *through i* (no look-ahead): schedule a forward bounce when the
        # oversold-rising trigger fires at bar i (dem_{i-1}<0.3 and dem_i>dem_{i-1}).
        if i >= period:
            sm_max = de_max_buf[i - period + 1:i + 1].sum()
            sm_min = de_min_buf[i - period + 1:i + 1].sum()
            denom = sm_max + sm_min
            dem_i = (sm_max / denom) if denom > 0 else 0.5
            if edge > 0.0 and dem_prev1 < 0.30 and dem_i > dem_prev1:
                end = min(i + 1 + bounce_days, n_days)
                bounce[i + 1:end] += per_day
            dem_prev2, dem_prev1 = dem_prev1, dem_i

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "period": period,
             "bounce_days": bounce_days, "n_days": n_days, "seed": seed}
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
