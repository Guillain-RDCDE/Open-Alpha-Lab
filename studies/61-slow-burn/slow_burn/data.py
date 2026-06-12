"""Data for the leveraged-ETF study — an offline synthetic price world, and a cached TQQQ/QQQ pair."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    vol_ann: float

    @property
    def has_vol(self) -> bool:
        return self.vol_ann > 0.0


def synthetic_underlying(n_years=15, drift=0.10, vol_ann=0.20, seed=61):
    """A daily underlying-return series — deterministic given ``seed``. ``vol_ann`` controls the
    volatility drag a leveraged version will suffer (0 ⇒ no drag, the null)."""
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2010-01-01", periods=n, name="date")
    r = drift / TRADING_DAYS + (vol_ann / np.sqrt(TRADING_DAYS)) * rng.standard_normal(n)
    return pd.Series(r, index=idx, name="underlying"), WorldTruth(vol_ann)


def fetch_pair(cache_dir=DEFAULT_CACHE, fetch=False):
    """Daily total returns for TQQQ (3x) and QQQ (1x), cache-first."""
    cache = os.path.join(cache_dir, "slow_burn_pair.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(["TQQQ", "QQQ"], period="max", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().dropna()
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
