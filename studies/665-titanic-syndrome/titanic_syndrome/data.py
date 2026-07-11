"""Data layer for Study 665 — Titanic Syndrome.

Bill Ohama's 1965 rule (as described by SentimenTrader / StockCharts / mcoscillator.com,
the same secondary literature the Hindenburg-Omen studies on this desk cite): the market
must print a fresh **52-week high** within the past **seven trading sessions**, and on
one of those (or a following) session the count of stocks making fresh **52-week LOWS**
must **exceed** the count making fresh 52-week highs. A strong tape throwing up more
breakdowns than breakouts, right after a high-water mark, is read as an internal warning
that the rally lacks confirmation — "the band is playing while the ship is already
listing."

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily adjusted closes for the **current 30 Dow Jones Industrial Average
  members** (the breadth basket — a coarse, survivorship-biased proxy for the thousands
  of NYSE-listed issues the original rule was built on; named honestly, see
  ``docs/references.md``) plus daily ^GSPC (price index, for the "near a 52-week high"
  context, per the brief) and SPY total-return closes (for forward returns and the
  timer), all from yfinance (no key), cached as CSV under the study's own ``_cache/``.
  We use each stock's own trailing 252-session high/low as its "52-week extreme" — the
  standard operational proxy (the original NYSE tape isn't free data); the index's fresh
  52-week high stands in for Ohama's "new high in equities" (a literal *all-time* high
  is not computable from a panel that starts in 2008 without pretending we have data we
  don't — this is stated as a named simplification, not hidden).

* **Synthetic world.** A deterministic, seeded panel of ``n_stocks`` i.i.d. log-normal
  price paths plus an equal-weight index, with a TUNABLE planted negative drift in the
  30 sessions following a genuine Titanic-Syndrome cluster (knob ``crash_bps``). At
  ``crash_bps = 0`` signal days carry zero information — the detector must not
  manufacture significance out of it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
DOW_CACHE = os.path.join(CACHE_DIR, "ts_dow30.csv")
GSPC_CACHE = os.path.join(CACHE_DIR, "ts_gspc.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "ts_spy.csv")

START = "2008-06-01"          # after Visa's 2008-03-19 IPO, the youngest current Dow member
AS_OF = "2026-06-30"          # last complete month at publication (2026-07-10)

# --------------------------------------------------------------------------- #
# Breadth basket — current (2026) Dow Jones Industrial Average members.
# Hardcoded, current membership only: a coarse, SURVIVORSHIP-BIASED proxy for the
# thousands of NYSE issues the original 1965 rule was built on. Named on the Signal
# axis (see docs/references.md) — names removed from the Dow over the sample window
# (GE, Pfizer, Intel, Walgreens, ExxonMobil, Raytheon, DowDuPont, ...) are excluded,
# which likely UNDERSTATES the true new-lows count during any of their idiosyncratic
# declines (the bias plausibly points AGAINST the signal, same direction named on the
# sibling 167-hindenburg-omen study's S&P panel).
# --------------------------------------------------------------------------- #
DOW30 = sorted([
    "MMM", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "VZ", "V", "WMT",
])
assert len(DOW30) == 30


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2007-01-01", end: str = "2026-07-01") -> None:
    """Download the Dow-30 panel, ^GSPC OHLC and SPY (total-return) closes; cache. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    raw = yf.download(DOW30, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes[DOW30].dropna(how="all")
    closes.to_csv(DOW_CACHE)

    gspc = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(gspc.columns, pd.MultiIndex):
        gspc.columns = gspc.columns.get_level_values(0)
    gspc[["Close"]].dropna().to_csv(GSPC_CACHE)

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy[["Close"]].dropna().to_csv(SPY_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (DOW_CACHE, GSPC_CACHE, SPY_CACHE))


def load_real(start: str = START, asof: str = AS_OF
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cached (dow30 panel, ^GSPC close, SPY close) frames, sliced to [start, asof]."""
    dow = pd.read_csv(DOW_CACHE, index_col=0, parse_dates=True).sort_index()
    gspc = pd.read_csv(GSPC_CACHE, index_col=0, parse_dates=True).sort_index()
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    out = []
    for df in (dow, gspc, spy):
        out.append(df.loc[(df.index >= start) & (df.index <= asof)].copy())
    return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Synthetic world — planted post-signal drift (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(crash_bps: float = 0.0, seed: int = 665,
                     n_stocks: int = 30, n_days: int = 4500,
                     daily_vol: float = 0.017, drift_bps: float = 3.2, rho: float = 0.40,
                     ath_window: int = 7, lookback: int = 252,
                     ) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic Dow-30-sized panel + equal-weight index with a TUNABLE planted
    post-signal drift.

    ``n_stocks`` price paths share a one-factor correlation structure (pairwise
    correlation ``rho`` ~ 0.40, the realistic order for large-cap Dow-style names —
    i.i.d. paths would make "half the panel prints highs, half prints lows"
    spuriously common and is NOT a faithful null) around a common market factor,
    building an equal-weight index (business-day index, ~18 years — far below the
    250-year pandas ns-timestamp trap). The Ohama rule (index at a fresh
    ``lookback``-day high within the past ``ath_window`` sessions, AND that day's
    new-lows count exceeds new-highs across the panel) is detected on this null
    panel; when a cluster fires, an EXTRA ``crash_bps`` of daily negative drift is
    injected into the following 30 sessions of the index return series.
    ``crash_bps = 0`` is the null world: signal days carry no information and the
    detector must not manufacture significance out of it.

    Returns (panel of daily returns wide-form incl. an ``index`` column, the boolean
    Titanic-signal Series aligned to the panel).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days)
    mu = drift_bps * 1e-4
    mkt = rng.standard_normal(n_days)
    idio = rng.standard_normal((n_days, n_stocks))
    shock = np.sqrt(rho) * mkt[:, None] + np.sqrt(1.0 - rho) * idio
    log_ret = mu - 0.5 * daily_vol ** 2 + daily_vol * shock
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    n_highs = np.zeros(n_days, dtype=int)
    n_lows = np.zeros(n_days, dtype=int)
    for t in range(lookback, n_days):
        window = prices[t - lookback:t, :]
        px = prices[t, :]
        n_highs[t] = int((px >= window.max(axis=0)).sum())
        n_lows[t] = int((px <= window.min(axis=0)).sum())

    index_px = prices.mean(axis=1)
    idx_roll_max = pd.Series(index_px).rolling(lookback, min_periods=lookback).max().to_numpy()
    at_high = index_px >= idx_roll_max - 1e-9
    near_high = pd.Series(at_high).rolling(ath_window, min_periods=1).max().astype(bool).to_numpy()
    lows_exceed = n_lows > n_highs
    signal = near_high & lows_exceed & (np.arange(n_days) >= lookback)

    index_ret = np.zeros(n_days)
    index_ret[1:] = index_px[1:] / index_px[:-1] - 1.0

    if crash_bps != 0:
        penalty = -crash_bps * 1e-4
        for cd in np.where(signal)[0]:
            for fwd in range(1, 31):
                if cd + fwd < n_days:
                    index_ret[cd + fwd] += penalty

    panel = pd.DataFrame({
        "n_highs": n_highs, "n_lows": n_lows, "n_stocks": n_stocks,
        "index_close": index_px, "index_ret": index_ret, "near_high": near_high,
    }, index=idx)
    return panel, pd.Series(signal, index=idx, name="titanic")
