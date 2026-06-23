"""Data layer for Study 394 — Defense-Basket (defense names/ETF + SPY, around shocks).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a defense basket (LMT, RTX, NOC, GD, plus the
  iShares U.S. Aerospace & Defense ETF ``ITA``) and the market proxy ``SPY`` (yfinance, no
  key), cached under ``_cache/defense_prices.csv`` (a wide CSV indexed by date, one column
  per ticker). From these we build a **defense basket** — the equal-weight average daily
  return of the four single names — and measure its return *in excess of SPY* over short
  windows following a fixed, hardcoded set of **geopolitical-shock dates** (war / invasion /
  rearmament headlines). The shock dates are the event study's treatment.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable defense
  excess-return series with an event-window edge *injected by construction* on a known set of
  shock dates. It is the positive control: the engine must recover a planted edge, and — with
  the edge knob set to zero — must NOT manufacture significance from a couple-dozen events.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "defense_prices.csv")

# The defense basket: four long-listed prime contractors (equal weight) + the sector ETF as a
# cross-check. These are explicit, named tickers; ITA (iShares U.S. Aerospace & Defense) only
# lists from 2006, so the single-name basket carries the long history and ITA is a robustness
# cross-check on its own (shorter) window.
DEFENSE_NAMES = ["LMT", "RTX", "NOC", "GD"]   # equal-weight defense basket (long history)
DEFENSE_ETF = "ITA"                           # iShares U.S. Aerospace & Defense (from 2006)
MARKET = "SPY"

# A fixed, hardcoded set of geopolitical-shock / war / rearmament headline dates — the
# believers' "defense rallies on war news" treatment. Each is the first trading day on or
# after a widely-reported shock; the list is transparent and editable (see docs/references.md
# for sourcing). NOT cherry-picked on outcome — picked on the *headline*, ex-ante.
SHOCK_DATES = [
    "1998-08-20",  # US strikes on Sudan/Afghanistan (Operation Infinite Reach)
    "1998-12-16",  # Operation Desert Fox (Iraq bombing)
    "1999-03-24",  # NATO begins bombing Yugoslavia (Kosovo)
    "2001-09-17",  # First trading day after 9/11
    "2001-10-08",  # US invades Afghanistan (Enduring Freedom)
    "2003-03-20",  # US invades Iraq
    "2008-08-08",  # Russia–Georgia war
    "2011-03-21",  # Western intervention in Libya
    "2014-03-03",  # Russia annexes Crimea (markets reopen)
    "2014-09-23",  # US-led airstrikes on ISIS in Syria
    "2015-09-30",  # Russia enters Syria
    "2017-04-07",  # US Tomahawk strike on Syria (Shayrat)
    "2018-04-13",  # US/UK/France strikes on Syria
    "2019-09-16",  # Abqaiq drone attack on Saudi oil (markets reopen)
    "2020-01-03",  # Soleimani killing (US–Iran escalation)
    "2021-08-16",  # Fall of Kabul / Afghanistan withdrawal
    "2022-02-24",  # Russia invades Ukraine
    "2023-10-09",  # First trading day after Hamas attack on Israel (Oct 7)
    "2024-04-15",  # Iran's direct missile/drone strike on Israel (markets reopen)
    "2024-10-01",  # Iran's second ballistic-missile barrage on Israel
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1995-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the defense names + ETF + SPY via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/defense_prices.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    tickers = [MARKET] + DEFENSE_NAMES + [DEFENSE_ETF]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def defense_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the daily-return frame the strategy consumes.

    Returns a frame indexed by date with columns:
      * ``mkt``     — SPY daily simple return
      * ``basket``  — equal-weight daily return of the four defense names
      * ``excess``  — ``basket - mkt`` (defense return in excess of the market)
      * ``etf``     — ITA daily return (NaN before 2006; a robustness cross-check)

    Days where fewer than two basket names have a return are dropped (early gaps).
    """
    rets = prices.pct_change()
    names = [c for c in DEFENSE_NAMES if c in rets.columns]
    basket = rets[names].mean(axis=1, skipna=True)
    enough = rets[names].notna().sum(axis=1) >= 2
    basket = basket.where(enough)
    mkt = rets[MARKET] if MARKET in rets.columns else pd.Series(index=rets.index, dtype=float)
    etf = rets[DEFENSE_ETF] if DEFENSE_ETF in rets.columns else pd.Series(np.nan, index=rets.index)
    out = pd.DataFrame({"mkt": mkt, "basket": basket, "excess": basket - mkt, "etf": etf})
    out = out.dropna(subset=["mkt", "basket"])
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> defense daily-return frame in one call."""
    return defense_frame(load_prices(path))


def shock_dates() -> pd.DatetimeIndex:
    """The hardcoded geopolitical-shock event dates (as a DatetimeIndex)."""
    return pd.DatetimeIndex(pd.to_datetime(SHOCK_DATES))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_defense(n_days: int = 7_000, n_events: int = 20,
                      edge_window: float = 0.0, seed: int = 394,
                      mkt_mu: float = 0.0003, mkt_sig: float = 0.011,
                      beta: float = 0.85, idio_sig: float = 0.012,
                      window: int = 21) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Deterministic defense frame with ``n_events`` shocks and a PLANTED window edge.

    Builds a market return series and a defense basket as ``beta*mkt + idiosyncratic noise``
    (so ``excess`` has *zero* expected value by construction). Then carves ``n_events``
    non-overlapping shock dates; if ``edge_window`` != 0 an *extra* cumulative excess return
    of ``edge_window`` is spread across the ``window`` trading days following each shock — the
    planted edge the event study must recover.

    With ``edge_window = 0`` the control must NOT find significance: ~20 events cannot reject
    the null however the noise falls — the whole small-sample lesson, on data where we know
    the truth.
    """
    rng = np.random.default_rng(seed)
    mkt = rng.normal(mkt_mu, mkt_sig, size=n_days)
    idio = rng.normal(0.0, idio_sig, size=n_days)
    basket = beta * mkt + idio
    excess = basket - mkt

    span = n_days // (n_events + 2)
    ev_pos = []
    for k in range(n_events):
        s = span * (k + 1) + int(rng.integers(0, max(1, span // 4)))
        ev_pos.append(s)
        if edge_window != 0.0:
            lo, hi = s, min(s + window, n_days)
            per_day = edge_window / window
            excess[lo:hi] += per_day
            basket[lo:hi] += per_day        # the edge shows up in the basket too

    # period_range keeps a decorative date label without OutOfBounds risk for long series
    idx = pd.period_range("1990-01-01", periods=n_days, freq="D").to_timestamp()
    frame = pd.DataFrame({"mkt": mkt, "basket": basket, "excess": excess,
                          "etf": np.nan}, index=idx)
    events = pd.DatetimeIndex([idx[p] for p in ev_pos])
    return frame, events
