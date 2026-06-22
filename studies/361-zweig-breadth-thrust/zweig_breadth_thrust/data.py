"""Data layer for Study 361 — Zweig Breadth Thrust (breadth proxy + SPY).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for SPY and a large basket of liquid US large-caps
  (yfinance, no key), cached under ``_cache/basket_prices.csv`` (a wide CSV indexed by date,
  one column per ticker plus ``SPY``). From the basket we build a **breadth proxy**: the
  fraction of constituents that closed *up* on each day (``adv_ratio``). True NYSE
  advance/decline breadth is not freely available, so this is an explicit, transparent
  proxy of the same object Zweig used.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable
  breadth series with thrust events *injected by construction* followed by a known forward
  drift. It is the positive control: the detector must find the injected thrusts and the
  inference must recover the planted edge (and, with the edge set to zero, must NOT
  manufacture significance out of a dozen events).

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "basket_prices.csv")

# A transparent, fixed basket of large, long-listed US large-caps used as the breadth
# proxy. Chosen for long price history and sector spread; this is a PROXY for true NYSE
# breadth, named as such everywhere. (Survivorship is acknowledged on the Signal axis: a
# fixed surviving-names basket is breadth-of-the-survivors, a mild upward bias.)
BASKET = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "GE", "IBM",
    "CVX", "PFE", "MRK", "T", "VZ", "INTC", "CSCO", "HD", "MCD", "DIS",
    "BA", "CAT", "MMM", "HON", "UNH", "WFC", "C", "BAC", "ORCL", "PEP",
    "ABT", "TXN", "COST", "LOW", "AMGN", "GS", "USB", "AXP", "DE", "DUK",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "1995-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + the basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/basket_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = ["SPY"] + BASKET
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    # keep names present for >=60% of the window (long-listed survivors)
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.60]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers + SPY)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def breadth_proxy(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the breadth proxy from basket prices.

    Returns a frame indexed by date with columns:
      * ``spy``       — SPY adjusted close
      * ``adv_ratio`` — fraction of basket constituents that closed up that day
                        (advancers / (advancers + decliners); unchanged excluded)

    This is the proxy for Zweig's NYSE (advancing issues / total issues). Days where fewer
    than 10 basket names have data are dropped (early years before the basket fills in).
    """
    cols = [c for c in prices.columns if c != "SPY"]
    basket = prices[cols]
    spy = prices["SPY"]
    chg = basket.diff()
    adv = (chg > 0).sum(axis=1)
    dec = (chg < 0).sum(axis=1)
    total = adv + dec
    enough = total >= 10
    adv_ratio = (adv / total).where(enough)
    out = pd.DataFrame({"spy": spy, "adv_ratio": adv_ratio})
    out = out.dropna()
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> breadth-proxy frame in one call."""
    return breadth_proxy(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_breadth(n_days: int = 7_000, n_events: int = 12,
                      edge_6m: float = 0.0, seed: int = 361,
                      mu_daily: float = 0.0003, sig_daily: float = 0.010) -> pd.DataFrame:
    """Deterministic breadth series with ``n_events`` thrusts injected by construction.

    Builds a daily ``adv_ratio`` mean-reverting around 0.5, then carves ``n_events``
    non-overlapping *thrust windows*: ``adv_ratio`` is forced below 0.40 then ramped above
    0.615 within ~10 days, exactly the Zweig pattern. A SPY-like price series is generated
    with daily drift ``mu_daily`` and vol ``sig_daily``; if ``edge_6m`` != 0 an *extra*
    cumulative drift of ``edge_6m`` is spread over the ~126 trading days following each
    injected thrust (the planted forward edge the detector+inference must recover).

    With ``edge_6m = 0`` the control must NOT find significance: ~12 events cannot reject
    the null however the noise falls — that is the whole sample-size lesson, demonstrated on
    data where we *know* the truth.
    """
    rng = np.random.default_rng(seed)
    # mean-reverting breadth around 0.5 (AR(1)-ish), clipped to (0.05, 0.95)
    ar = np.empty(n_days)
    ar[0] = 0.5
    for t in range(1, n_days):
        ar[t] = 0.5 + 0.80 * (ar[t - 1] - 0.5) + rng.normal(0, 0.06)
    adv = np.clip(ar, 0.05, 0.95)

    # daily SPY-like log-returns
    ret = rng.normal(mu_daily, sig_daily, size=n_days)

    # inject non-overlapping thrust windows + (optionally) a planted forward edge.
    # Pattern that makes the EMA(span=10) cross 0.615 having been <0.40: hold breadth
    # deeply low for ~6 days, then hold it high for ~12 days so the smoothed EMA pulls up
    # through both thresholds exactly once.
    span = n_days // (n_events + 2)
    for k in range(n_events):
        s = span * (k + 1) + int(rng.integers(0, span // 4))
        lo_days, hi_days = 6, 12
        adv[s:s + lo_days] = rng.uniform(0.18, 0.30, size=lo_days)
        end = s + lo_days + hi_days
        if end > n_days:
            continue
        adv[s + lo_days:end] = rng.uniform(0.82, 0.92, size=hi_days)
        if edge_6m != 0.0:
            h = 126                            # spread the planted edge over ~6 months
            lo, hi = end, min(end + h, n_days)
            ret[lo:hi] += edge_6m / h
    adv = np.clip(adv, 0.05, 0.95)

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    return pd.DataFrame({"spy": price, "adv_ratio": adv}, index=idx)
