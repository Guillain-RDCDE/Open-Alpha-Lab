"""Data for the betting-against-beta study — an offline synthetic cross-section, and a cached ETF panel.

  * :func:`synthetic_cross_section` — **offline, deterministic**. A market factor plus N assets with
    betas spread across a range; low-beta assets carry a small ``low_beta_premium`` alpha (the effect)
    — set it to 0 for the null. Daily returns. Pins the BAB machinery offline.
  * :func:`fetch_etf_panel` — a real cross-section of liquid ETFs spanning the beta spectrum, SPY as
    the market, **and the 13-week T-bill yield (^IRX)** so every Sharpe in the study can be quoted
    excess-of-cash and the self-financing ledger has a real funding rate. Daily, **cache-first**.
    Fingerprinted run in ``docs/results.md``.

One structural caveat the strategy module leans on: the "low-beta half" of this 13-ETF panel is
usually **TLT/IEF/GLD plus the defensive sectors** — bonds and gold, not just quiet stocks. Frazzini &
Pedersen build BAB *within* an asset class; a cross-asset panel makes the long leg substantially a
levered duration/gold book. ``EQUITY_ONLY`` (the panel minus TLT/IEF/GLD) exists so the study can show
both versions side by side.
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

# Liquid ETFs spanning low→high beta, + SPY as the market, + ^IRX as the cash rate.
ETFS = ["XLU", "XLP", "XLV", "TLT", "IEF", "GLD", "XLK", "XLY", "XLF", "XLE", "XLI", "IWM", "EEM"]
MARKET = "SPY"
TBILL = "^IRX"  # 13-week T-bill discount yield, percent annualised
# The intra-equity subset — drop the bond/gold tickers so BAB sorts within one asset class,
# the way Frazzini & Pedersen actually build it.
NON_EQUITY = ["TLT", "IEF", "GLD"]
EQUITY_ONLY = [t for t in ETFS if t not in NON_EQUITY]


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
    ``(assets_daily, market_daily, truth)``. (No cash rate: the synthetic world tests the machinery
    with ``rf = 0``.)
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
                    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Daily returns of the liquid-ETF cross-section + SPY, plus the T-bill yield, cache-first.

    **Cache-only** unless ``fetch=True``. Returns ``(assets_daily, market_daily, rf_ann_daily)``
    aligned from 1999, where ``rf_ann_daily`` is the ^IRX 13-week discount yield as an annualised
    *fraction* (e.g. 0.045), forward-filled — the cash rate behind every excess-of-cash Sharpe and the
    base of the financing ledger. The in-progress calendar month is dropped at fetch time so the last
    month is always complete. Returns ``(empty, empty, empty)`` on a cache miss with ``fetch=False``.
    """
    cache = os.path.join(cache_dir, "free_lunch_etf_panel.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        if "rf_ann" in df.columns:
            return df[ETFS], df[MARKET], df["rf_ann"]
        # old cache schema (no ^IRX) → treat as a miss and refetch
    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    import yfinance as yf  # lazy

    raw = yf.download(ETFS + [MARKET, TBILL], period="max", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    ret = raw[ETFS + [MARKET]].pct_change()
    rf = (raw[TBILL].ffill() / 100.0).rename("rf_ann")
    df = ret.join(rf).loc["1999-01-01":].dropna(how="all")
    # never let a few days of the in-progress month masquerade as a monthly return downstream
    last_complete = pd.Timestamp.today().to_period("M") - 1
    df = df[df.index.to_period("M") <= last_complete]
    df.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache)
    return df[ETFS], df[MARKET], df["rf_ann"]


def monthly_rf(rf_ann_daily: pd.Series) -> pd.Series:
    """Monthly cash return from the daily annualised T-bill yield.

    Month *t* earns the yield as of the **prior** month-end divided by 12 — the rate you could
    actually lock walking into the month, no look-ahead.
    """
    return (pd.Series(rf_ann_daily).astype(float).resample("ME").last().ffill() / 12.0).shift(1).rename("rf")
