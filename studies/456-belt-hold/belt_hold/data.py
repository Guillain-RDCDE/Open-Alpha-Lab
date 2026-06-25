"""Data layer for Study 456 (Belt-Hold / opening marubozu).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A bullish belt-hold is a bar that **opens at its low** (no lower wick) and **closes well up**,
  arriving after a downtrend; the believers' claim is that the open-at-the-extreme marks a
  reversal and the next days drift **up**. We plant exactly that: with ``edge > 0`` a bar that
  is *recognised as a bullish belt-hold by the rule* (open == low, big white body, prior
  downtrend) is followed by a real upward push over the next few sessions; with ``edge = 0`` the
  log-return series is a pure random walk and the belt-hold flag is a fair coin. This is the
  positive control — a harness that cannot bank the planted reversal proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the belt-hold flag is
read on the close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs candlestick proponents draw on: the broad tape, big-cap tech, small caps, and
# a couple of cross-asset charts. Daily, liquid, long history.
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
    seed: int = 456,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of belt-hold reversal.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Bars are then dressed with opens and wicks. On a deterministic subset of bars we *manufacture*
    a bullish belt-hold (open exactly at the low, a tall white body) and, when ``edge > 0`` and a
    real prior downtrend is present, we add an upward drift over the following ``trend_lookback``
    sessions proportional to ``edge``. At ``edge = 0`` the belt-hold bars carry no such drift and
    the flag is a fair coin; at ``edge > 0`` a recognised bullish belt-hold is followed by a real
    bounce the detector should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Base close path: pure random walk (no drift) in log space.
    eps = rng.normal(0.0, daily_vol, n_days)
    # We will inject post-belt-hold pushes, so build the path incrementally.
    log_c = np.empty(n_days)
    lp = np.log(100.0)

    # Pre-decide which bars are "belt-hold candidates" (sparse, deterministic): ~6% of bars.
    is_candidate = rng.random(n_days) < 0.06
    # A planted upward push that decays over the next trend_lookback bars after a recognised
    # bullish belt-hold. We add it to the eps of the following bars.
    push = np.zeros(n_days)

    body = np.empty(n_days)  # signed body size in log units (close-open), filled below
    body_white = rng.random(n_days) > 0.5  # otherwise random direction for non-candidates

    for i in range(n_days):
        lp += eps[i] + push[i]
        log_c[i] = lp
        # On a candidate bar, check whether a *real* prior downtrend exists (close fell over the
        # last trend_lookback bars). If so and edge>0, plant a forward push.
        if edge > 0.0 and is_candidate[i] and i >= trend_lookback:
            prior = log_c[i] - log_c[i - trend_lookback]
            if prior < 0.0:  # genuine downtrend into the belt-hold
                mag = edge * daily_vol * 6.0
                for j in range(1, trend_lookback + 1):
                    if i + j < n_days:
                        push[i + j] += mag * (1.0 - j / (trend_lookback + 1.0))

    close = np.exp(log_c)

    # Opens: usually previous close, but on candidate bars we force open == low (belt-hold shape).
    open_ = np.empty(n_days)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Default wicks.
    wick_hi = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    wick_lo = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick_hi
    lo = np.minimum(open_, close) - wick_lo

    # Dress the candidate bars into bullish belt-holds: open == low, tall white body, no lower wick.
    for i in range(n_days):
        if is_candidate[i]:
            c = close[i]
            tall = daily_vol * 2.2 * c
            o = c - tall                 # white body: open below close
            open_[i] = o
            lo[i] = o                    # open exactly at the low -> no lower wick
            hi[i] = c + abs(rng.normal(0.0, daily_vol * 0.15, 1)[0]) * c  # tiny/zero upper wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
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
