"""Data for the idiosyncratic-volatility study — an offline synthetic panel, and a cached real panel.

  * :func:`synthetic_panel` — **offline, deterministic**. Stocks load on a common market factor; each
    carries a fixed idiosyncratic volatility, and high-idio-vol stocks get a drift penalty of strength
    ``idiovol_premium`` — so high-idio-vol names underperform, by construction. 0 = the null.
  * :func:`fetch_panel` — daily returns for current S&P 500 members + the market (SPY), **cache-first**.
    Survivorship-biased, large-cap (stated). Fingerprinted run in ``docs/results.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    idiovol_premium: float

    @property
    def has_premium(self) -> bool:
        return self.idiovol_premium != 0.0


def synthetic_panel(
    n_stocks: int = 120, n_years: int = 22, idiovol_premium: float = 0.0008, mkt_vol: float = 0.16,
    seed: int = 54
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A daily stock panel with an idio-vol structure — deterministic given ``seed``.

    A market factor drives all stocks (beta 1); each stock has a fixed idiosyncratic volatility spread
    over [low, high], and high-idio-vol stocks get a per-day drift penalty ``-idiovol_premium·rank`` —
    so the high-idio-vol names underperform. ``idiovol_premium = 0`` is the null. Returns
    ``(daily_returns, market, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2003-01-01", periods=n, name="date")
    mkt = (mkt_vol / np.sqrt(TRADING_DAYS)) * rng.standard_normal(n)
    rank = np.linspace(0.0, 1.0, n_stocks)
    rng.shuffle(rank)
    iv = (0.10 + 0.40 * rank) / np.sqrt(TRADING_DAYS)     # idio vol spread, high rank → high idio-vol
    drift = 0.0006 - idiovol_premium * rank               # high idio-vol → low drift
    R = drift[None, :] + mkt[:, None] + iv[None, :] * rng.standard_normal((n, n_stocks))
    cols = [f"S{j:03d}" for j in range(n_stocks)]
    return (pd.DataFrame(R, index=idx, columns=cols),
            pd.Series(mkt, index=idx, name="SPY"), WorldTruth(idiovol_premium))


def fetch_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, min_days: int = TRADING_DAYS * 15
                ) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns of current S&P 500 members + the market (SPY), cache-first.

    **Cache-only** unless ``fetch=True``. Returns ``(daily, market)``; survivorship-biased and
    large-cap — the idio-vol puzzle is documented strongest in small/illiquid names.
    """
    pcache = os.path.join(cache_dir, "static_panel.parquet")
    mcache = os.path.join(cache_dir, "static_market.parquet")
    if os.path.exists(pcache) and os.path.exists(mcache):
        return pd.read_parquet(pcache), pd.read_parquet(mcache)["SPY"]
    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float)
    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.universe import sp500_symbols
    import yfinance as yf

    syms = sp500_symbols()
    px = yf.download(syms + ["SPY"], period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.pct_change().loc["1999-01-01":]
    keep = [c for c in ret.columns if c != "SPY" and ret[c].dropna().shape[0] >= min_days]
    daily, mkt = ret[keep], ret["SPY"]
    daily.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    daily.to_parquet(pcache)
    mkt.to_frame("SPY").to_parquet(mcache)
    return daily, mkt
