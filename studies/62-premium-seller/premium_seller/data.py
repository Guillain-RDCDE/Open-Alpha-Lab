"""Data for the covered-call study — an offline synthetic world, and a cached QYLD/QQQ/SPY panel."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TICKERS = ["QYLD", "QQQ", "SPY"]


@dataclass(frozen=True)
class WorldTruth:
    cap: float

    @property
    def is_capped(self) -> bool:
        return self.cap < 1.0


def synthetic_world(n_years=12, cap=0.5, premium=0.005, mkt_vol=0.18, seed=62):
    """A monthly covered-call world: QYLD = QQQ with up-moves scaled by ``cap`` (<1 caps upside) plus a
    monthly ``premium``. ``cap = 1`` and ``premium = 0`` is the null (no covered-call distortion)."""
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("2014-01-31", periods=n, freq="ME", name="date")
    qqq = (mkt_vol / np.sqrt(12)) * rng.standard_normal(n) + 0.010
    qyld = np.where(qqq > 0, cap * qqq, qqq) + premium
    spy = 0.85 * qqq + (0.06 / np.sqrt(12)) * rng.standard_normal(n)
    return pd.DataFrame({"QYLD": qyld, "QQQ": qqq, "SPY": spy}, index=idx), WorldTruth(cap)


def fetch_panel(cache_dir=DEFAULT_CACHE, fetch=False):
    """Monthly total returns for QYLD, QQQ, SPY, cache-first."""
    cache = os.path.join(cache_dir, "premium_seller_panel.parquet")
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
