"""Data for the commodity-skewness study — an offline synthetic panel, and a cached commodity-ETF panel."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252
ETFS = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "CORN", "WEAT", "SOYB", "PPLT", "PALL", "DBC", "GSG", "UGA"]


@dataclass(frozen=True)
class WorldTruth:
    skew_premium: float

    @property
    def has_premium(self) -> bool:
        return self.skew_premium != 0.0


def synthetic_panel(n_assets=14, n_years=18, skew_premium=0.0010, vol_ann=0.25, seed=60):
    """A daily commodity panel with a skewness structure — deterministic given ``seed``.

    Each asset gets a fixed lottery score; high scorers receive rare positive jumps (de-meaned, so
    mean-neutral but positively skewed) *and* a drift penalty ``-skew_premium*score`` — so high-skew
    assets underperform. ``skew_premium = 0`` is the null. Returns a (date x asset) daily-return frame.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2008-01-01", periods=n, name="date")
    score = rng.uniform(0, 1, n_assets)
    sig = vol_ann / np.sqrt(TRADING_DAYS)
    drift = 0.0003 - skew_premium * score
    R = drift[None, :] + sig * rng.standard_normal((n, n_assets))
    # rare positive jumps for high-score assets, de-meaned -> positive skew, mean-neutral
    jp = 0.004 * (0.2 + score)
    jsize = 0.08
    jumps = (rng.random((n, n_assets)) < jp[None, :]) * jsize
    R = R + jumps - (jp * jsize)[None, :]
    cols = [f"C{j:02d}" for j in range(n_assets)]
    return pd.DataFrame(R, index=idx, columns=cols), WorldTruth(skew_premium)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False, min_days=TRADING_DAYS * 8):
    """Daily returns of a commodity-ETF basket, cache-first."""
    cache = os.path.join(cache_dir, "long_shot_panel.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(ETFS, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().loc["2008-01-01":]
    keep = [c for c in ret.columns if ret[c].dropna().shape[0] >= min_days]
    ret = ret[keep]
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
