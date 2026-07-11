"""Data layer for Study 642 — Turnaround Tuesday.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily SPY OHLC + adjusted (total-return) close from yfinance (no key),
  cached as CSV under the study's own ``_cache/``. We use the **adjusted** close for
  every return calculation (total-return, dividends folded in) — this is a close-to-close
  calendar study, not an intraday one, so there is no raw-vs-adjusted geometry question
  the way there is for a high-low range.

* **No hardcoded event calendar.** Unlike a scheduled-announcement study (FOMC, earnings),
  "Monday" and "Tuesday" are calendar facts read straight off the trading-day index
  (``.dayofweek``) — there is nothing to hardcode. The only "calendar" fact this study
  needs is which calendar day of the week each session falls on, which pandas already
  knows.

* **Synthetic world.** A deterministic, seeded i.i.d.-drift daily-return tape with a
  TUNABLE planted "turnaround" bounce (knob ``bounce``, in return units): whenever a
  Monday session closes down, the following Tuesday session gets an extra ``+bounce``
  tacked onto its return. ``bounce = 0`` is the null world — down-Monday Tuesdays
  statistically identical to up-Monday Tuesdays; the Welch machinery must NOT
  manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "tt_spy.csv")

START = "1993-01-29"        # SPY inception
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
ERA_SPLIT = "2000-01-01"    # pre/post-2000 era split named in the brief


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1993-01-01", end: str = "2026-07-01") -> None:
    """Download SPY raw OHLC + adjusted close; cache it. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    spy = pd.DataFrame({
        "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
        "Close": raw["Close"],
        "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
    }).dropna(how="all")
    spy.to_csv(SPY_CACHE)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE)


def load_real(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached SPY frame, sliced to [start, asof]."""
    df = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


# --------------------------------------------------------------------------- #
# Synthetic world — planted down-Monday -> Tuesday bounce (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bounce: float = 0.0, seed: int = 642,
                    start: str = "1993-02-01", end: str = "2026-06-30",
                    daily_vol: float = 0.010, drift: float = 0.08 / 252,
                    ) -> pd.DataFrame:
    """Deterministic i.i.d.-drift daily-return tape with a TUNABLE planted bounce.

    Business-day index (no holidays — a synthetic control, not a calendar-accuracy
    test), so every Monday is immediately followed by a Tuesday session. On a Monday
    that closes down (its own drawn return < 0), the following Tuesday's return gets
    an EXTRA ``bounce`` added (the planted turnaround effect). ``bounce = 0`` is the
    null world: down-Monday and up-Monday Tuesdays are statistically identical, and
    the Welch split must NOT reach significance.

    Business-day index, span ~33 years — far below the 250-year pandas ns-timestamp
    trap. Returns a one-column ``Close`` frame.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    dow = idx.dayofweek.to_numpy()

    rets = rng.normal(drift, daily_vol, n)
    mon_idx = np.where(dow == 0)[0]
    for t in mon_idx:
        if t + 1 < n and dow[t + 1] == 1 and rets[t] < 0:
            rets[t + 1] += bounce                  # the planted turnaround bounce

    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Close": close}, index=idx)
