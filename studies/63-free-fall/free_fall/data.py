"""Data for the short-vol study — an offline synthetic carry world, and a cached SVXY/SPY pair."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252
TICKERS = ["SVXY", "SPY"]


@dataclass(frozen=True)
class WorldTruth:
    crash_prob: float

    @property
    def has_crash_risk(self) -> bool:
        return self.crash_prob > 0.0


def synthetic_shortvol(n_years=10, carry=0.0010, day_vol=0.02, crash_prob=0.002, crash_size=0.6, seed=63):
    """A daily short-vol return series — deterministic given ``seed``. A steady ``carry`` with small
    noise most days, plus rare catastrophic crashes (prob ``crash_prob``, size ~``crash_size``) — the
    steamroller. ``crash_prob = 0`` is the null (carry with no tail). Returns ``(shortvol, market)``."""
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2011-01-03", periods=n, name="date")
    r = carry + day_vol * rng.standard_normal(n)
    crash = rng.random(n) < crash_prob
    r = np.where(crash, -crash_size * (0.5 + rng.random(n)), r)
    mkt = 0.0004 + (0.18 / np.sqrt(TRADING_DAYS)) * rng.standard_normal(n)
    return pd.DataFrame({"SVXY": r, "SPY": mkt}, index=idx), WorldTruth(crash_prob)


def fetch_pair(cache_dir=DEFAULT_CACHE, fetch=False):
    """Daily total returns for SVXY (short-vol) and SPY, cache-first."""
    cache = os.path.join(cache_dir, "free_fall_pair.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf
    px = yf.download(TICKERS, period="max", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().dropna()
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret
