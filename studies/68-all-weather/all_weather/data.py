"""Data for the risk-parity study — an offline synthetic multi-asset world, and the cached cross-asset
ETF panel (SPY / IEF / GLD / DBC)."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
ASSETS = ["SPY", "IEF", "GLD", "DBC"]          # stocks, Treasuries, gold, commodities


@dataclass(frozen=True)
class WorldTruth:
    vol_spread: float

    @property
    def vols_differ(self) -> bool:
        return self.vol_spread != 0.0


def synthetic_world(n_days=4000, vol_spread=0.012, seed=68, n_assets=4):
    """A daily multi-asset world. Each asset has the same Sharpe but different volatility (spread set by
    ``vol_spread``); assets are lightly correlated. Risk parity should then lift portfolio Sharpe and cut
    drawdown vs equal-weight. ``vol_spread=0`` ⇒ all vols equal ⇒ risk parity ≈ equal weight (the null)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n_days, name="date")
    cols = ["SPY", "IEF", "GLD", "DBC"][:n_assets]
    base_vol = 0.006
    vols = base_vol + vol_spread * np.arange(n_assets)          # 0, +spread, +2·spread, ...
    sharpe_d = 0.03                                             # same daily Sharpe for all
    common = rng.standard_normal(n_days)                        # a light common factor
    data = {}
    for k, c in enumerate(cols):
        idio = rng.standard_normal(n_days)
        shock = 0.3 * common + 0.95 * idio
        data[c] = sharpe_d * vols[k] + vols[k] * shock
    return pd.DataFrame(data, index=idx), WorldTruth(vol_spread)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Daily returns for SPY / IEF / GLD / DBC, cache-first (from the shared ``cross_asset_etfs`` pull)."""
    cache = os.path.join(cache_dir, "cross_asset_etfs.parquet")
    if os.path.exists(cache):
        px = pd.read_parquet(cache)[ASSETS].dropna()
        px.index = pd.DatetimeIndex(px.index).tz_localize(None)
        ret = px.pct_change().dropna(); ret.index.name = "date"
        return ret
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(ASSETS, period="max", auto_adjust=True, progress=False)["Close"][ASSETS].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().dropna(); ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    px.to_parquet(os.path.join(cache_dir, "cross_asset_etfs.parquet"))
    return ret
