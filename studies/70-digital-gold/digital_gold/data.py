"""Data for the bitcoin "digital gold" study — an offline synthetic world, and the cached BTC / SPY /
GLD daily panel."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
ASSETS = ["BTC-USD", "SPY", "GLD"]


@dataclass(frozen=True)
class WorldTruth:
    btc_stock_beta: float

    @property
    def is_risk_asset(self) -> bool:
        return self.btc_stock_beta > 0.0


def synthetic_world(n_days=3000, btc_stock_beta=1.5, seed=70):
    """A daily world with stocks, gold and a bitcoin-like asset. Gold is ~uncorrelated with stocks; BTC
    loads on the stock factor with ``btc_stock_beta`` (and its own high vol). ``btc_stock_beta>0`` ⇒ BTC
    behaves like a (high-beta) risk asset that falls *with* stocks; ``=0`` ⇒ a true uncorrelated haven."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-09-18", periods=n_days, name="date")
    eq_shock = rng.standard_normal(n_days)
    spy = 0.0005 + 0.011 * eq_shock
    gld = 0.0003 + 0.010 * rng.standard_normal(n_days)                      # ~independent of stocks
    btc = 0.001 + btc_stock_beta * 0.011 * eq_shock + 0.012 * rng.standard_normal(n_days)
    df = pd.DataFrame({"BTC-USD": btc, "SPY": spy, "GLD": gld}, index=idx)
    return df, WorldTruth(btc_stock_beta)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Daily returns for BTC-USD / SPY / GLD, cache-first (from the ``digital_gold_panel`` pull)."""
    cache = os.path.join(cache_dir, "digital_gold_panel.parquet")
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
    os.makedirs(cache_dir, exist_ok=True)
    px.to_parquet(cache)
    ret = px.pct_change().dropna(); ret.index.name = "date"
    return ret
