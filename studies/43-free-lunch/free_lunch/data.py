"""Data for the betting-against-beta study — an offline synthetic cross-section, and a cached ETF panel.

  * :func:`synthetic_cross_section` — **offline, deterministic**. A market factor plus N assets with
    betas spread across a range; low-beta assets carry a small ``low_beta_premium`` alpha (the effect)
    — set it to 0 for the null. Daily returns. Pins the BAB machinery offline.
  * :func:`fetch_etf_panel` — a real cross-section of liquid ETFs spanning the beta spectrum + SPY as
    the market, daily, **cache-first**. Fingerprinted run in ``docs/results.md``.
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

# Liquid ETFs spanning low→high beta, + SPY as the market.
ETFS = ["XLU", "XLP", "XLV", "TLT", "IEF", "GLD", "XLK", "XLY", "XLF", "XLE", "XLI", "IWM", "EEM"]
MARKET = "SPY"


@dataclass(frozen=True)
class WorldTruth:
    low_beta_premium: float

    @property
    def has_premium(self) -> bool:
        return self.low_beta_premium != 0.0


def synthetic_cross_section(
    n_assets: int = 12, n_years: int = 25, low_beta_premium: float = 0.04,
    mkt_vol: float = 0.16, idio_vol: float = 0.12, seed: int = 43
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A factor world for BAB — deterministic given ``seed``.

    A market factor (annual vol ``mkt_vol``) drives ``n_assets`` whose betas are spread evenly over
    [0.4, 1.8]. Each asset earns ``beta·market + alpha + idio``, where ``alpha`` is
    ``low_beta_premium·(1 − beta)`` annualised — so low-beta names out-earn their beta when the premium
    is on, and the BAB book should profit gross. ``low_beta_premium = 0`` is the null. Returns
    ``(assets_daily, market_daily, truth)``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS
    idx = pd.bdate_range("2000-01-03", periods=n, name="date")
    betas = np.linspace(0.4, 1.8, n_assets)
    mkt = (mkt_vol / np.sqrt(TRADING_DAYS)) * rng.standard_normal(n)
    alpha_d = low_beta_premium * (1.0 - betas) / TRADING_DAYS
    idio = (idio_vol / np.sqrt(TRADING_DAYS)) * rng.standard_normal((n, n_assets))
    assets = betas[None, :] * mkt[:, None] + alpha_d[None, :] + idio
    cols = [f"A{j:02d}_b{betas[j]:.1f}" for j in range(n_assets)]
    return (pd.DataFrame(assets, index=idx, columns=cols),
            pd.Series(mkt, index=idx, name="market"), WorldTruth(low_beta_premium))


def fetch_etf_panel(cache_dir: str = DEFAULT_CACHE, fetch: bool = False
                    ) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns of the liquid-ETF cross-section + SPY, cache-first.

    **Cache-only** unless ``fetch=True``. Returns ``(assets_daily, market_daily)`` aligned from 1999,
    or ``(empty, empty)`` on a cache miss with ``fetch=False``.
    """
    cache = os.path.join(cache_dir, "free_lunch_etf_panel.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df[ETFS], df[MARKET]
    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float)
    import yfinance as yf  # lazy

    raw = yf.download(ETFS + [MARKET], period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    ret = raw.pct_change().loc["1999-01-01":].dropna(how="all")
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_parquet(cache)
    return ret[ETFS], ret[MARKET]
