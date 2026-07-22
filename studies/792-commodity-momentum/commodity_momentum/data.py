"""Data layer for Study 792 — Cross-sectional commodity momentum.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily total-return closes (yfinance ``auto_adjust=True`` — distributions
  folded in) for a basket of **13 liquid single-commodity ETFs** spanning metals, energy
  and agriculturals::

      GLD SLV PPLT PALL CPER   USO UNG UGA   CORN WEAT SOYB CANE DBA

  cached as one parquet under the study's own ``_cache/``. Monthly returns are the
  month-end-to-month-end pct-change of those adjusted closes. This is the **investable**
  reading of the Miffre-Rallis claim: a real portfolio you could hold is a basket of
  single-commodity ETFs, not un-rollable continuous-futures quotes.

  **Survivorship note (named on the Signal axis).** The basket is *current* membership —
  every fund still trades in 2026. Commodity ETFs do close (leveraged/niche ones
  especially), so a current-membership basket is mildly biased *upward* (a fund that
  trended to closure is absent from the loser leg). These are broad, liquid, physically- or
  front-future-backed single-commodity funds that rarely delist, so the bias is small — but
  it points *for* the effect, so any positive momentum number here is an upper bound.

* **Synthetic world.** A deterministic, seeded monthly return panel with a TUNABLE planted
  cross-sectional momentum effect (knob ``mom_edge``): each asset carries a persistent
  latent trend (AR(1)); observed return = i.i.d. noise + ``mom_edge`` x latent trend.
  Because the trend is persistent, past-year winners are the high-latent assets and they
  keep winning — exactly the effect a 12-1 sort is built to harvest. ``mom_edge = 0`` is
  the null world (pure noise, no momentum); the sort must NOT manufacture a significant
  L/S return from it.

Pure numpy + pandas + stdlib on the offline path. :func:`fetch` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
ETF_CACHE = os.path.join(CACHE_DIR, "cm_etf_daily.parquet")

# 13 liquid single-commodity ETFs — metals / energy / agriculturals.
TICKERS = [
    "GLD", "SLV", "PPLT", "PALL", "CPER",     # precious + base metals
    "USO", "UNG", "UGA",                       # energy
    "CORN", "WEAT", "SOYB", "CANE", "DBA",     # agriculturals
]

# Human-readable underlying, for the notebooks.
UNDERLYING = {
    "GLD": "gold", "SLV": "silver", "PPLT": "platinum", "PALL": "palladium",
    "CPER": "copper", "USO": "WTI crude", "UNG": "natural gas", "UGA": "gasoline",
    "CORN": "corn", "WEAT": "wheat", "SOYB": "soybeans", "CANE": "sugar",
    "DBA": "ag basket",
}

START = "2005-01-01"        # GLD inception is 2004-11; basket fills in by ~2012
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07)
ERA_SPLIT = "2019-01-01"    # ~midpoint of the effective (post-2012) monthly sample


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download the basket's total-return daily closes; cache one parquet. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame({s: raw[s]["Close"] for s in TICKERS
                              if s in raw.columns.get_level_values(0)})
    else:  # single-ticker fallback (never used with 13 names, kept for safety)
        close = raw[["Close"]].rename(columns={"Close": TICKERS[0]})
    close = close.dropna(how="all").sort_index()
    close.to_parquet(ETF_CACHE)


def have_real() -> bool:
    return os.path.exists(ETF_CACHE)


def load_prices(start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily total-return closes, sliced to [start, asof]."""
    df = pd.read_parquet(ETF_CACHE).sort_index()
    df.index = pd.to_datetime(df.index)
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


def monthly_returns(prices: pd.DataFrame | None = None,
                    asof: str = AS_OF, min_live: int = 6) -> pd.DataFrame:
    """Month-end total-return matrix (one column per ETF).

    Resample the daily adjusted closes to month-end last, pct-change to monthly returns,
    drop the (possibly partial) as-of month if it is not a true month-end, and keep only
    months where at least ``min_live`` assets have a return (so the cross-sectional sort
    always ranks a real breadth of names).
    """
    if prices is None:
        prices = load_prices(asof=asof)
    m_last = prices.resample("ME").last()
    mret = m_last.pct_change()
    # Drop a trailing partial month (as-of not a real month-end close).
    asof_ts = pd.Timestamp(asof)
    mret = mret.loc[mret.index <= asof_ts]
    if len(mret) and mret.index[-1].to_period("M") == asof_ts.to_period("M") \
            and asof_ts != asof_ts.to_period("M").to_timestamp("M"):
        mret = mret.iloc[:-1]
    live = mret.notna().sum(axis=1)
    return mret.loc[live >= min_live]


# --------------------------------------------------------------------------- #
# Synthetic world — planted cross-sectional momentum (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(mom_edge: float = 0.0, seed: int = 792, n_assets: int = 13,
                    n_months: int = 168, vol: float = 0.06,
                    phi: float = 0.9, trend_vol: float = 0.03) -> pd.DataFrame:
    """Deterministic monthly return panel with a TUNABLE planted momentum effect.

    Each asset carries a persistent latent trend ``L[i,t]`` following an AR(1)
    (``L[t] = phi*L[t-1] + shock``); the observed monthly return is
    ``noise + mom_edge * L[i,t]``. Because the trend is persistent, an asset's past-year
    cumulative return is informative about its *current* trend, so a 12-1 winner keeps
    winning — the exact structure a cross-sectional momentum sort harvests.

    ``mom_edge = 0`` is the null world: returns are i.i.d. noise, past winners carry no
    information, and the L/S sort must NOT produce a significant mean. Monthly index,
    ~14 years — far below the pandas ns-timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.0, trend_vol, (n_months, n_assets))
    latent = np.zeros((n_months, n_assets))
    latent[0] = shock[0]
    for t in range(1, n_months):
        latent[t] = phi * latent[t - 1] + shock[t]
    noise = rng.normal(0.0, vol, (n_months, n_assets))
    rets = noise + mom_edge * latent
    idx = pd.period_range("2012-01", periods=n_months, freq="M").to_timestamp("M")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(rets, index=idx, columns=cols)
