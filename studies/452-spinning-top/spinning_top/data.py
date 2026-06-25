"""Data layer for Study 452 (Spinning-Top).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A spinning top is a small-bodied candle with comparable upper and lower wicks — the textbook
  "indecision" bar. The believers' claim is that this indecision *resolves into a directional
  move*. We plant exactly that: with ``edge > 0`` a spinning-top day is **followed by a burst of
  upward drift** (the indecision "resolves up"), so a spinning-top entry harvests a real move;
  with ``edge = 0`` the post-pattern return is drawn from the same distribution as any other day
  and the entry is a fair coin. This is the positive control — a harness that cannot bank the
  planted resolution proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a spinning top is
classified on the close of *t* (open/high/low/close all known on that bar), and the trade is
entered at *t+1*'s close.
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
# a couple of cross-asset charts. Daily, liquid, long history (reused so it runs offline).
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 452,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of post-spinning-top drift.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Each bar gets a random body size and two wicks; with probability ~12% we draw a deliberate
    **spinning top** (tiny body, two long comparable wicks). The planted force lives in the
    *forward* return: on the bars **following** a spinning top we inject an extra upward drift
    proportional to ``edge`` (the indecision "resolves up"). At ``edge = 0`` a spinning-top day
    is followed by the same return distribution as any other day, so the entry is a fair coin;
    at ``edge > 0`` a spinning-top entry banks a real directional move the detector should find.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # 1) decide which days are spinning tops (independent of price, so edge=0 is a fair coin)
    is_top = rng.random(n_days) < 0.12

    # 2) build the close path; inject planted drift on the day AFTER a spinning top
    base_ret = rng.normal(0.0, daily_vol, n_days)
    bonus = np.zeros(n_days)
    if edge > 0.0:
        # the day after a spinning top gets an extra upward push (the "resolution")
        prev_top = np.concatenate([[False], is_top[:-1]])
        bonus[prev_top] = edge * daily_vol
    rets = base_ret + bonus
    log_p = np.log(100.0) + np.cumsum(rets)
    close = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # 3) shape each candle's body + wicks
    hi = np.empty_like(close)
    lo = np.empty_like(close)
    for i in range(n_days):
        o, c = open_[i], close[i]
        body_top = max(o, c)
        body_bot = min(o, c)
        body = body_top - body_bot
        ref = close[i]
        if is_top[i]:
            # spinning top: tiny body, two LONG comparable wicks
            wick = (0.6 + 0.4 * rng.random()) * daily_vol * ref
            up_w = wick * (0.85 + 0.30 * rng.random())
            dn_w = wick * (0.85 + 0.30 * rng.random())
        else:
            # ordinary bar: short, often lopsided wicks
            up_w = abs(rng.normal(0.0, daily_vol * 0.35)) * ref
            dn_w = abs(rng.normal(0.0, daily_vol * 0.35)) * ref
        hi[i] = body_top + up_w
        lo[i] = body_bot - dn_w
        # guarantee a planted top really reads as small-body if the random body was large
        if is_top[i] and body > 0:
            rng_span = hi[i] - lo[i]
            if rng_span > 0 and body / rng_span > 0.22:
                # widen wicks so the body is < ~22% of range
                pad = (body / 0.18 - rng_span) / 2.0
                hi[i] += pad
                lo[i] -= pad

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days,
             "seed": seed, "n_tops": int(is_top.sum())}
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
