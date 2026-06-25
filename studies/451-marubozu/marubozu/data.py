"""Data layer for Study 451 (Marubozu).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**
  specific to THIS indicator. A bullish marubozu is a session whose body fills almost the whole
  range (open ≈ low, close ≈ high). The believers' claim is *continuation*: after a wickless
  up-day, the next days keep rising. We plant exactly that: with ``edge > 0`` the day **after**
  a (synthetically generated) wickless up-bar receives a small extra upward drift, so a
  marubozu-entry harvests a real continuation; with ``edge = 0`` wicks and bodies are drawn
  independently of the next return, so a marubozu is a fair coin. This is the positive control —
  a harness that cannot bank the planted continuation proves nothing by finding nothing on the
  real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a cached
  parquet if present and only touches the network on an explicit cache miss (with a short
  back-off + retry), then caches the parquet so re-runs are offline. Daily history is long
  (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a marubozu is read on
the close of *t* (its own OHLC, fully known at the close), and the trade is entered at *t+1*'s
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

# Liquid, long-history indices/ETFs candlestick traders draw marubozus on: the broad tape,
# big-cap tech, small caps, the Dow, and a cross-asset chart. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 451,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of marubozu-continuation.

    Each session is a body (open→close) plus two wicks. We draw a body return and two
    *non-negative* wick lengths; on a random subset of up-days we **shrink both wicks to ~0**,
    manufacturing genuine bullish marubozus. The planted force is **continuation**: with
    ``edge > 0`` the bar *after* a wickless up-bar gets an extra upward drift proportional to
    ``edge``; with ``edge = 0`` the wick-shrink is independent of the next return, so a marubozu
    carries no information and the entry is a fair coin.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    open_ = np.empty(n_days)
    close = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    # Draw the *whole* zero-drift body-return path first (so the tape itself has no net drift).
    # A bar becomes a bullish marubozu when its body return is up AND it is flagged "wickless".
    body_ret = rng.normal(0.0, daily_vol, n_days)  # zero-drift base returns
    wickless = rng.random(n_days) < 0.12           # ~12% of bars are drawn with (near) no wicks

    # Continuation force: a marubozu (wickless up-bar) pulls the NEXT bar up by edge*sigma. With
    # edge=0 this is identically zero, so the marubozu day carries no info -> fair coin.
    px = 100.0
    is_maru = np.zeros(n_days, dtype=bool)
    carry_left = 0                                  # days of continuation drift still owed
    for i in range(n_days):
        o = px
        drift = edge * daily_vol if carry_left > 0 else 0.0
        if carry_left > 0:
            carry_left -= 1
        r = body_ret[i] + drift
        c = o * np.exp(r)

        body_hi, body_lo = max(o, c), min(o, c)
        if wickless[i]:
            wick_up = abs(rng.normal(0.0, daily_vol * 0.02)) * o
            wick_dn = abs(rng.normal(0.0, daily_vol * 0.02)) * o
            if c > o:                               # a wickless UP-bar = a bullish marubozu
                is_maru[i] = True
                if edge > 0.0:                      # plant a multi-day continuation drift
                    carry_left = 10
        else:
            wick_up = abs(rng.normal(0.0, daily_vol * 0.6)) * o
            wick_dn = abs(rng.normal(0.0, daily_vol * 0.6)) * o

        open_[i] = o
        close[i] = c
        hi[i] = body_hi + wick_up
        lo[i] = body_lo - wick_dn
        px = c

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days,
             "seed": seed, "planted_maru": int(is_maru.sum())}
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
