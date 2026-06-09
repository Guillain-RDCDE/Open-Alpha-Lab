"""Data access for the stat-arb-optimization study — the panel and where it comes from.

This study trades a short-horizon **cross-sectional mean reversion**: each stock's idiosyncratic
residual tends to bounce the next day, and an "optimal" portfolio harvests it. So the tape is a panel of
daily returns, and the data layer keeps the desk's offline/cache split:

    * :func:`synthetic_panel` — fully **offline**. Each stock is ``r_i = beta_i*market + e_{i,t} +
      idio*eps``, where the idiosyncratic part ``e`` is an AR(1) with a **negative** coefficient
      (``-revert``) — i.e. it *mean-reverts* one day to the next. So yesterday's residual predicts a
      reversal in today's: the signal the optimizer trades. ``revert = 0`` is the **null** (white-noise
      residual, nothing to revert). Deterministic given ``seed``.
    * :func:`fetch_panel` — the current S&P 500 cross-section via :mod:`quantlab.universe`, reduced to a
      daily-returns panel. **Cache-only** unless ``fetch=True``; the network import stays lazy.

Same data choices as the other panel studies: total-return closes; current-membership universe
(survivorship bias — ranking robust, magnitudes not). The residual is taken against the equal-weight
market (1-factor), and the covariance is the *sample* covariance — whose instability is the whole point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PanelTruth:
    n_stocks: int
    n_bars: int
    revert: float             # one-day reversion strength of the residual (AR(1) coef = -revert); 0 == null

    @property
    def has_reversion(self) -> bool:
        return self.revert != 0.0


def synthetic_panel(n_stocks: int = 60, n_bars: int = 3024, revert: float = 0.20,
                    mkt_drift: float = 0.0003, mkt_vol: float = 0.010, idio: float = 0.012,
                    seed: int = 0) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A panel whose idiosyncratic residual **mean-reverts** one day to the next, by construction.

    ``r_{i,t} = beta_i*market[t] + e_{i,t}``, ``e_{i,t} = -revert*e_{i,t-1} + idio*eps`` (negative AR(1)
    ⇒ reversion). Yesterday's residual predicts today's reversal — the stat-arb signal. ``revert = 0``
    gives a white-noise residual (the null). Returns ``(panel, market, truth)``; deterministic given
    ``seed``. A modest ``n_stocks`` keeps the sample covariance estimable (the instability the study is
    about appears as the universe grows relative to the lookback).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2012-01-02", periods=n_bars, name="date")
    market = pd.Series(mkt_drift + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.7)
    e = np.zeros((n_bars, n_stocks))
    eps = rng.standard_normal((n_bars, n_stocks))
    for t in range(1, n_bars):
        e[t] = -revert * e[t - 1] + idio * eps[t]
    rets = betas[None, :] * market.to_numpy()[:, None] + e
    panel = pd.DataFrame(rets, index=idx, columns=[f"STK{i:02d}" for i in range(n_stocks)])
    return panel, market, PanelTruth(n_stocks=n_stocks, n_bars=n_bars, revert=revert)


def fetch_panel(start: str = "2010-01-01", min_days: int = 2500, max_names: int = 80,
                fetch: bool = False, cache_dir: str | None = None) -> pd.DataFrame:
    """Current S&P 500 daily-returns panel via :mod:`quantlab.universe`, cache-first.

    Capped at ``max_names`` (the most-complete histories) so the sample covariance is estimable — the
    point of the study is what happens to ``C^{-1}`` as the universe grows, not to drown in 500 names.
    """
    import sys
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if cache_dir is not None:
        os.environ["OVERNIGHT_CACHE"] = cache_dir
    from quantlab import universe
    symbols = universe.sp500_symbols(use_cache=True)
    cache_file = os.path.join(os.environ.get("OVERNIGHT_CACHE", os.path.join(REPO_ROOT, "_cache")),
                              f"panel_{len(symbols)}_{start}.parquet")
    if not os.path.exists(cache_file) and not fetch:
        return pd.DataFrame()
    panel = universe.download_panel(symbols, start=start, use_cache=True)
    closes = {tk: o["Close"].dropna() for tk, o in panel.items() if len(o["Close"].dropna()) >= min_days}
    if not closes:
        return pd.DataFrame()
    rets = pd.DataFrame(closes).sort_index().pct_change().dropna(how="all")
    # keep the max_names with the longest histories (deterministic)
    keep = rets.notna().sum().sort_values(ascending=False).head(max_names).index
    rets = rets[sorted(keep)]
    rets.index.name = "date"
    return rets
