"""Data layer for Study 384 — ISM-PMI-Regime (manufacturing-PMI proxy + SPY).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for SPY and a fixed basket of large, long-listed
  US **industrial / manufacturing** names (yfinance, no key), cached under
  ``_cache/manuf_prices.csv`` (a wide CSV indexed by date, one column per ticker plus
  ``SPY``). The true ISM Manufacturing PMI is a **proprietary survey** (the Institute for
  Supply Management sells it) and the usual free fallback (FRED series ``NAPM`` / regional
  Fed diffusion indices) is **blocked from this environment**. So we build an explicit,
  transparent **PMI proxy**: a monthly *diffusion index* across the industrial basket —
  the share of constituents whose 3-month price momentum is positive — rescaled onto a
  PMI-like 0–100 axis centred at 50. A reading **> 50** is the proxy's "expansion" regime,
  exactly the threshold the folklore uses. It is a stand-in for the survey's diffusion of
  "new orders rising," not the survey itself, and is named a proxy everywhere.

* **Synthetic.** A deterministic, fixed-seed generator that produces a controllable
  PMI-like regime series and an SPY-like price whose drift differs by regime by a *known*
  ``edge`` knob. It is the positive control: with ``edge=0`` the regime split must NOT
  manufacture significance; with a large planted ``edge`` the test must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "manuf_prices.csv")

# A transparent, fixed basket of large, long-listed US industrial / manufacturing names
# used as the PMI proxy. Chosen for long price history and manufacturing-cycle exposure
# (machinery, aerospace/defence, autos, chemicals, materials, transports). This is a PROXY
# for the ISM Manufacturing PMI's diffusion of "new orders," named as such everywhere.
# (Survivorship is acknowledged on the Signal axis: a fixed surviving-names basket is the
# manufacturing breadth *of the survivors*, a mild upward bias on the regime.)
BASKET = [
    "GE", "CAT", "BA", "MMM", "HON", "DE", "EMR", "ITW", "ETN", "PH",
    "ROK", "DOV", "PCAR", "CMI", "GD", "LMT", "NOC", "RTX", "TXT", "SWK",
    "DD", "DOW", "PPG", "NUE", "X", "AA", "FCX", "VMC", "F", "GM",
    "UNP", "CSX", "NSC", "FDX", "UPS", "WM", "JCI", "IR", "PNR", "FAST",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "1995-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download SPY + the industrial basket via yfinance and cache a wide adj-close CSV.

    Network-only; used once to build ``_cache/manuf_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = ["SPY"] + BASKET
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    # keep names present for >=40% of the window (long-listed survivors)
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.40]
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


def pmi_proxy(prices: pd.DataFrame, lookback: int = 63, span: int = 3) -> pd.DataFrame:
    """Build the monthly PMI proxy + SPY from basket prices.

    Returns a **monthly** frame (month-end) indexed by date with columns:
      * ``spy`` — SPY adjusted close at month-end
      * ``pmi`` — the PMI proxy: a diffusion index on a 0–100 axis. Each month we measure,
                  across the industrial basket, the fraction of names whose trailing
                  ``lookback``-day (~3-month) return is positive (the breadth of rising
                  "new orders"), times 100. A reading > 50 means most of manufacturing is
                  expanding. Smoothed with a ``span``-month EMA, like the survey's inertia.

    This is the PROXY for the ISM Manufacturing PMI (diffusion of new orders). Months where
    fewer than 10 basket names have data are dropped (early years before the basket fills in).
    """
    cols = [c for c in prices.columns if c != "SPY"]
    basket = prices[cols]
    spy = prices["SPY"]
    # daily diffusion: share of names with positive trailing-`lookback` return
    mom = basket / basket.shift(lookback) - 1.0
    up = (mom > 0).sum(axis=1)
    have = mom.notna().sum(axis=1)
    enough = have >= 10
    diff = (up / have).where(enough)                       # 0..1
    pmi_daily = (diff * 100.0)
    # month-end sampling
    pmi_m = pmi_daily.resample("ME").last()
    spy_m = spy.resample("ME").last()
    pmi_m = pmi_m.ewm(span=span, adjust=False).mean()
    out = pd.DataFrame({"spy": spy_m, "pmi": pmi_m}).dropna()
    return out


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Convenience: cached prices -> monthly PMI-proxy + SPY frame in one call."""
    return pmi_proxy(load_prices(path))


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_regime(n_months: int = 600, edge: float = 0.0, seed: int = 384,
                     base_mu: float = 0.007, sig: float = 0.040,
                     pmi_persist: float = 0.85) -> pd.DataFrame:
    """Deterministic PMI-like regime series + SPY-like price with a KNOWN regime edge.

    The PMI proxy follows a persistent AR(1) around 50 (a manufacturing cycle that spends
    long stretches above/below the 50 line). The monthly SPY return has base drift
    ``base_mu`` plus an *extra* ``edge`` drift **only in months where PMI > 50** — the
    planted "own stocks only above 50" effect the regime test must recover.

    With ``edge = 0`` the regime carries no information: the above-50 and below-50 monthly
    means differ only by noise, and the test must NOT cross t = 2. With a large ``edge``
    (e.g. +1.5%/month) it must light up. The date index is a decorative monthly label, so
    we use ``period_range`` to avoid any out-of-bounds-Timestamp issues on long spans.
    """
    rng = np.random.default_rng(seed)
    # persistent PMI cycle around 50
    pmi = np.empty(n_months)
    pmi[0] = 50.0
    for t in range(1, n_months):
        pmi[t] = 50.0 + pmi_persist * (pmi[t - 1] - 50.0) + rng.normal(0, 3.0)
    pmi = np.clip(pmi, 30.0, 70.0)

    above = pmi > 50.0
    ret = rng.normal(base_mu, sig, size=n_months)
    ret = ret + np.where(above, edge, 0.0)                 # planted regime edge
    price = 100.0 * np.cumprod(1.0 + ret)

    idx = pd.period_range("1970-01", periods=n_months, freq="M").to_timestamp(how="end")
    return pd.DataFrame({"spy": price, "pmi": pmi}, index=idx)
