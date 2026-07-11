"""Data layer for Study 668 — Williams VIX Fix.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily OHLC for an eight-name basket — three broad index ETFs (SPY, QQQ,
  IWM, no survivorship — they hold whoever is in the index today) and five large,
  currently-listed single names (AAPL, MSFT, JPM, XOM, JNJ) chosen for decades of clean
  history, from yfinance (no key), split+dividend adjusted (``auto_adjust=True`` folds
  corporate actions into the price so OHLC stays internally consistent), cached as CSV
  under the study's own ``_cache/``. **Survivorship, named:** the five single names are
  picked *because* they are still trading today with a long clean tape — a basket built
  this way can only ever show what capitulation bounces look like in names that lived to
  tell about it. The three ETFs carry no such bias (whoever is in SPY/QQQ/IWM today is by
  construction the current index, not "the five stocks that happened to survive").

* **Synthetic world.** A deterministic, seeded daily OHLC generator with occasional violent
  one-day crashes (a wide low-wick, exactly what the VIX Fix is built to catch) and a
  TUNABLE planted post-crash bounce (knob ``bounce``, extra daily drift for the ``bounce_days``
  sessions after a crash). ``bounce = 0`` is the null world — crash days carry no forward
  information; the machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# Three broad index ETFs (no survivorship) + five long-history single names (survivorship
# named on the Signal axis: picked because they are still trading today).
TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "JPM", "XOM", "JNJ"]
ETF_TICKERS = {"SPY", "QQQ", "IWM"}

START = "2000-01-03"   # 26+ years, spans dot-com bust, GFC, 2020 crash, 2022 bear
AS_OF = "2026-06-30"    # last complete month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"wvf_{ticker}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(tickers: list[str] = TICKERS, start: str = "1999-06-01",
          end: str = "2026-07-01") -> None:
    """Download split+dividend adjusted daily OHLC for the basket; cache one CSV/ticker."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in tickers:
        raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close"]].dropna(how="all")
        df.to_csv(_cache_path(t))


def have_real(tickers: list[str] = TICKERS) -> bool:
    return all(os.path.exists(_cache_path(t)) for t in tickers)


def load_real(tickers: list[str] = TICKERS, start: str = START,
              asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached {ticker: OHLC frame}, each sliced to [start, asof]."""
    out = {}
    for t in tickers:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        out[t] = df.loc[(df.index >= start) & (df.index <= asof)].copy()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted post-crash bounce (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bounce: float = 0.0, seed: int = 668,
                    n_days: int = 6500, start: str = "2000-01-03",
                    daily_vol: float = 0.011, crash_prob: float = 1.0 / 90.0,
                    crash_size: float = 0.07, bounce_days: int = 10,
                    ) -> pd.DataFrame:
    """Deterministic daily OHLC with occasional crash days and a TUNABLE planted bounce.

    Baseline log returns are i.i.d. normal (``daily_vol``). On any day, independently with
    probability ``crash_prob`` a "capitulation" event fires: that day's return gets an extra
    ``-crash_size`` shock and the low is stretched further down (a wide low-wick — exactly
    the geometry the VIX Fix is designed to catch). For the following ``bounce_days``
    sessions, an extra ``+bounce`` drift is added to every name's return (the planted
    capitulation-bounce effect). ``bounce = 0``: crash days carry no forward information and
    the Welch/NW detector must NOT fire. Business-day index, span ~26 years — far below the
    ns-timestamp trap.

    Returns an OHLC frame with a boolean ``crash`` column flagging the event day itself.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    n = len(idx)

    is_crash = rng.random(n) < crash_prob
    ret = rng.normal(0.0, daily_vol, n)
    ret[is_crash] -= crash_size

    # Planted post-crash bounce: add `bounce` drift to each of the `bounce_days` sessions
    # following a crash (does not compound across overlapping crash clusters — a simple
    # additive knob is enough to prove the machinery is unbiased).
    if bounce != 0.0:
        boost = np.zeros(n)
        crash_idx = np.flatnonzero(is_crash)
        for i in crash_idx:
            lo, hi = i + 1, min(i + bounce_days, n - 1)
            if lo <= hi:
                boost[lo:hi + 1] += bounce
        ret = ret + boost

    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    # Ordinary session wick: small symmetric noise around the open-close range.
    wick = np.abs(rng.normal(0.0, daily_vol * 0.4, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    # On a crash day, stretch the low further (the intrabar panic the VIX Fix targets) —
    # independent of the close-to-close return itself, so a naive close-only drawdown proxy
    # cannot see the full extent of it.
    low = np.where(is_crash, low - crash_size * 0.6 * close, low)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "crash": is_crash},
        index=idx,
    )
