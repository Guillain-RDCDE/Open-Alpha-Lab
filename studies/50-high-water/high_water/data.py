"""Data for the 52-week-high study — an offline synthetic trending panel, and a cached stock panel.

  * :func:`synthetic_panel` — **offline, deterministic**. A cross-section of stocks each with a slowly
    switching trend (drift regime) of strength ``trend`` — so a stock that's been rising is near its
    high *and* has positive momentum *and* tends to keep rising (nearness and momentum both predict,
    and they're correlated, by construction). ``trend = 0`` is the null. Pins the machinery offline.
  * :func:`fetch_panel` — monthly returns for current S&P 500 members with ≥20y history (Yahoo),
    **cache-first**. Survivorship-biased, large-cap (stated). Fingerprinted run in ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
MONTHS = 12


@dataclass(frozen=True)
class WorldTruth:
    trend: float

    @property
    def has_trend(self) -> bool:
        return self.trend != 0.0


def synthetic_panel(
    n_stocks: int = 120, n_years: int = 26, trend: float = 0.10, vol_ann: float = 0.30,
    regime_months: float = 18.0, seed: int = 50
) -> tuple[pd.DataFrame, WorldTruth]:
    """A trending monthly stock panel — deterministic given ``seed``.

    Each stock follows a 3-state drift regime (up / flat / down) switching ~every ``regime_months``; in
    a trending regime the monthly drift is ``±trend·σ``, so a riser stays near its high and keeps rising
    — making nearness and momentum both predictive and mutually correlated. ``trend = 0`` is the null
    (iid, nothing predicts). Returns a (date × stock) monthly-return frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    sig = vol_ann / np.sqrt(MONTHS)
    p_switch = 1.0 / regime_months
    states = rng.choice([-1, 0, 1], size=n_stocks)
    R = np.empty((n, n_stocks))
    for t in range(n):
        flip = rng.random(n_stocks) < p_switch
        states = np.where(flip, rng.choice([-1, 0, 1], size=n_stocks), states)
        R[t] = trend * sig * states + sig * rng.standard_normal(n_stocks)
    cols = [f"S{j:03d}" for j in range(n_stocks)]
    return pd.DataFrame(R, index=idx, columns=cols), WorldTruth(trend)


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_months: int = 240) -> pd.DataFrame:
    """Monthly returns of current S&P 500 members with long history, cache-first.

    **Cache-only** unless ``fetch=True`` (Wikipedia membership + Yahoo monthly). Survivorship-biased
    and large-cap — stated, since 52-week-high effects are documented across the size spectrum and this
    is the tradable end.
    """
    cache = os.path.join(cache_dir, "high_water_panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    syms = sp500_symbols()
    px = yf.download(syms, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change()
    keep = [c for c in ret.columns if ret[c].dropna().shape[0] >= min_months]
    ret = ret[keep].loc["1995-01-01":]
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
