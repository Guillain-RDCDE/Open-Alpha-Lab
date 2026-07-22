"""Data layer for Study 793 — Cross-sectional commodity *value* (long-horizon reversal).

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily closes (yfinance) for the **same basket of 13 liquid
  single-commodity ETFs** used by sibling study 792 — metals, energy, agriculturals::

      GLD SLV PPLT PALL CPER   USO UNG UGA   CORN WEAT SOYB CANE DBA

  We keep **two** price series per name, cached as one parquet under the study's own
  ``_cache/`` (a column MultiIndex ``(ticker, field)`` with fields ``price`` and ``tr``):

  - ``price`` — the **raw, un-adjusted close** (``auto_adjust=False``). This is the
    *price level* the Asness-Moskowitz-Pedersen (AMP 2013) commodity **value** signal is
    built on: the ratio of the price ~5 years ago to the current price. Value is a
    *price-level* reversal claim, so the signal must be measured on the price, not on a
    total-return series that folds in five years of carry/roll.
  - ``tr`` — the **total-return (adjusted) close** (``Adj Close``, distributions folded
    in). This is what a held position actually *earns*, so the L/S book's monthly P&L is
    computed from ``tr``.

  **Survivorship note (named on the Signal axis).** The basket is *current* membership —
  every fund still trades in 2026. Commodity ETFs do close (leveraged/niche ones
  especially), so a current-membership basket is mildly biased. For a *value* (buy-the-
  fallen) sleeve the direction is subtle: a fund that fell so far it delisted is exactly a
  "deep value" name that never got to mean-revert, and its absence flatters the long
  (cheap) leg — so the value premium here is, if anything, an **upper bound**.

  **Roll-contamination caveat (named on the Signal axis).** For chronic-contango energy
  ETFs (USO, UNG, UGA) even the *raw* price grinds down over years from negative roll
  yield, so they read as perennially "cheap" for a mechanical reason that has nothing to
  do with a value reversal. The ETF proxy cannot fully separate spot-price value from roll
  drift the way AMP's spot series can — a real limitation that travels with the stamp.

* **Synthetic world.** A deterministic, seeded monthly **price** panel with a TUNABLE
  planted long-horizon reversal (knob ``val_edge``): each asset's next monthly return
  carries a drift = ``val_edge`` x (log price ~5y ago − log price now), i.e. names that
  have fallen over five years are pushed back up. ``val_edge = 0`` is a pure random walk
  (no reversal); the value sort must NOT manufacture a significant L/S mean from it.

Pure numpy + pandas + stdlib on the offline path. :func:`fetch` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
ETF_CACHE = os.path.join(CACHE_DIR, "cv_etf_daily.parquet")

# The SAME 13 liquid single-commodity ETFs as sibling study 792.
TICKERS = [
    "GLD", "SLV", "PPLT", "PALL", "CPER",     # precious + base metals
    "USO", "UNG", "UGA",                       # energy
    "CORN", "WEAT", "SOYB", "CANE", "DBA",     # agriculturals
]

UNDERLYING = {
    "GLD": "gold", "SLV": "silver", "PPLT": "platinum", "PALL": "palladium",
    "CPER": "copper", "USO": "WTI crude", "UNG": "natural gas", "UGA": "gasoline",
    "CORN": "corn", "WEAT": "wheat", "SOYB": "soybeans", "CANE": "sugar",
    "DBA": "ag basket",
}

START = "2005-01-01"        # GLD inception 2004-11; basket fills in by ~2012
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07)
# A 5-year value signal only trades once every name carries 5y of history (~2016+), so
# the effective monthly sample is short; the era split is its midpoint.
ERA_SPLIT = "2019-07-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download the basket's raw + total-return daily closes; cache one parquet.

    Network; runs once. Stores a ``(ticker, field)`` MultiIndex-column frame with
    ``field`` in {``price`` (raw Close), ``tr`` (Adj Close)}.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    frames = {}
    for s in TICKERS:
        if isinstance(raw.columns, pd.MultiIndex) and s in raw.columns.get_level_values(0):
            sub = raw[s]
            close = sub["Close"]
            adj = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
            frames[(s, "price")] = close
            frames[(s, "tr")] = adj
    out = pd.DataFrame(frames).dropna(how="all").sort_index()
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["ticker", "field"])
    out.to_parquet(ETF_CACHE)


def have_real() -> bool:
    return os.path.exists(ETF_CACHE)


def _load_field(field: str, start: str, asof: str) -> pd.DataFrame:
    df = pd.read_parquet(ETF_CACHE).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.loc[(df.index >= start) & (df.index <= asof)]
    cols = [c for c in df.columns if c[1] == field]
    out = df[cols].copy()
    out.columns = [c[0] for c in cols]
    return out


def load_price(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily **raw** closes (the price level for the value signal), sliced."""
    return _load_field("price", start, asof)


def load_totalreturn(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily **total-return** (adjusted) closes (the P&L series), sliced."""
    return _load_field("tr", start, asof)


def _monthly_last(daily: pd.DataFrame, asof: str) -> pd.DataFrame:
    """Month-end last observation, dropping a trailing partial (non-month-end) month."""
    m = daily.resample("ME").last()
    asof_ts = pd.Timestamp(asof)
    m = m.loc[m.index <= asof_ts]
    if len(m) and m.index[-1].to_period("M") == asof_ts.to_period("M") \
            and asof_ts != asof_ts.to_period("M").to_timestamp("M"):
        m = m.iloc[:-1]
    return m


def monthly_price(asof: str = AS_OF) -> pd.DataFrame:
    """Month-end **raw price** levels (one column per ETF) — input to the value signal."""
    return _monthly_last(load_price(asof=asof), asof)


def monthly_returns(asof: str = AS_OF, min_live: int = 6) -> pd.DataFrame:
    """Month-end **total-return** matrix (one column per ETF) — the L/S book's P&L.

    Keep only months where at least ``min_live`` assets have a return so the
    cross-sectional sort always ranks a real breadth of names.
    """
    m_last = _monthly_last(load_totalreturn(asof=asof), asof)
    mret = m_last.pct_change()
    live = mret.notna().sum(axis=1)
    return mret.loc[live >= min_live]


# --------------------------------------------------------------------------- #
# Synthetic world — planted long-horizon reversal (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(val_edge: float = 0.0, seed: int = 793, n_assets: int = 13,
                    n_months: int = 168, vol: float = 0.06,
                    lookback: int = 60) -> pd.DataFrame:
    """Deterministic monthly **price** panel with a TUNABLE planted value reversal.

    Prices evolve as a log random walk whose next monthly return carries a drift equal to
    ``val_edge`` x (log price ``lookback`` months ago − log price now). A name that has
    fallen a lot over ~5 years therefore has a *positive* expected next return: it is
    pushed back toward where it was — exactly the long-horizon reversal a value sort is
    built to harvest. ``val_edge = 0`` is a pure random walk: past 5-year losers carry no
    information and the sort must NOT produce a significant L/S mean.

    Returns a price-level panel (monthly index, ~14 years) — the value signal reads the
    prices, the L/S P&L reads their ``pct_change``.
    """
    rng = np.random.default_rng(seed)
    logp = np.zeros((n_months, n_assets))
    logp[0] = rng.normal(0.0, 0.2, n_assets)          # dispersed starting levels
    for t in range(1, n_months):
        shock = rng.normal(0.0, vol, n_assets)
        if t > lookback:
            drift = val_edge * (logp[t - lookback] - logp[t - 1])
        else:
            drift = 0.0
        logp[t] = logp[t - 1] + drift + shock
    prices = 100.0 * np.exp(logp)
    idx = pd.period_range("2012-01", periods=n_months, freq="M").to_timestamp("M")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)
