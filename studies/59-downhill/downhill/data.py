"""Data for the term-premium study — an offline synthetic world, and a cached Treasury-ETF panel."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TICKERS = ["IEF", "SHY", "BIL"]   # 7-10y, 1-3y, cash


@dataclass(frozen=True)
class WorldTruth:
    premium: float

    @property
    def has_premium(self) -> bool:
        return self.premium != 0.0


def synthetic_world(n_years=22, premium=0.02, long_vol=0.07, seed=59):
    """A monthly Treasury world: cash (BIL, tiny vol), 1-3y (SHY), 7-10y (IEF). IEF earns ``premium``
    over cash but with duration volatility ``long_vol`` (so the excess can be poorly paid). 0 = null."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2003-01-31", periods=n, freq="ME", name="date")
    bil = 0.0015 + 0.002 / np.sqrt(12) * rng.standard_normal(n)
    shy = bil + (premium * 0.4) / 12 + (long_vol * 0.25) / np.sqrt(12) * rng.standard_normal(n)
    ief = bil + premium / 12 + long_vol / np.sqrt(12) * rng.standard_normal(n)
    return pd.DataFrame({"IEF": ief, "SHY": shy, "BIL": bil}, index=idx), WorldTruth(premium)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Monthly total returns for IEF, SHY, BIL, cache-first."""
    cache = os.path.join(cache_dir, "downhill_panel.parquet")
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
