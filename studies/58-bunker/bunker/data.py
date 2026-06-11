"""Data for the min-vol study — an offline synthetic world, and a cached ETF pair (USMV/SPY)."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TICKERS = ["USMV", "SPY"]


@dataclass(frozen=True)
class WorldTruth:
    lowvol_alpha: float

    @property
    def has_alpha(self) -> bool:
        return self.lowvol_alpha != 0.0


def synthetic_world(n_years=14, lowvol_alpha=0.0, beta=0.7, mkt_vol=0.15, idio_vol=0.05, seed=58):
    """A monthly USMV/SPY world: the min-vol sleeve has beta<1 (so lower vol) plus an optional
    ``lowvol_alpha`` (0 = the null, ~the real result: lower risk, no Sharpe edge)."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2011-11-30", periods=n, freq="ME", name="date")
    mkt = (mkt_vol / np.sqrt(12)) * rng.standard_normal(n) + 0.008
    usmv = beta * mkt + lowvol_alpha / 12.0 + (idio_vol / np.sqrt(12)) * rng.standard_normal(n)
    spy = mkt + (idio_vol / np.sqrt(12)) * rng.standard_normal(n)
    return pd.DataFrame({"USMV": usmv, "SPY": spy}, index=idx), WorldTruth(lowvol_alpha)


def fetch_pairs(cache_dir=DEFAULT_CACHE, fetch=False):
    """Monthly total returns for USMV and SPY, cache-first."""
    cache = os.path.join(cache_dir, "bunker_pairs.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(TICKERS, period="max", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
