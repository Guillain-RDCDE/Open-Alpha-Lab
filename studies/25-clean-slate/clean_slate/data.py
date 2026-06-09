"""Data access for the residual-momentum study — the panel and where it comes from.

Residual momentum is ordinary momentum run on the part of each stock's return that is *not* explained by
the market (its idiosyncratic residual). So the tape is a panel of daily returns, and the data layer
keeps the desk's offline/cache split:

    * :func:`synthetic_panel` — fully **offline**. Each stock is ``r_i = beta_i*market + theta_{i,t} +
      idio*eps``, where ``theta`` is a slow, *persistent* idiosyncratic drift (the AR(1) that makes a
      stock's *residual* — not its total return — keep trending). The market itself is volatile and can
      lurch, so **total**-return momentum picks up beta dispersion (and the crash that comes with it),
      while **residual** momentum sees only the clean, persistent ``theta``. ``mom_strength`` sets the
      drift amplitude; ``mom_strength = 0`` is the null. Deterministic given ``seed``.
    * :func:`fetch_panel` — the current S&P 500 cross-section via :mod:`quantlab.universe`, reduced to a
      daily-returns panel. **Cache-only** unless ``fetch=True``; the network import stays lazy.

Same two data choices as [Study 24](../../24-stampede/): total-return closes; current-membership
universe (survivorship bias — ranking robust, magnitudes not). The residual is taken against the
**equal-weight market** (a 1-factor residual); the source strategy uses Fama-French 3-factor residuals,
a stated simplification (we lack SMB/HML here).
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
    mom_strength: float
    phi: float

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


def synthetic_panel(n_stocks: int = 150, n_bars: int = 4032, mom_strength: float = 0.0016,
                    phi: float = 0.985, mkt_drift: float = 0.0003, mkt_vol: float = 0.011,
                    idio: float = 0.012, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A panel where the *residual* (not the total return) carries the persistent momentum.

    ``r_{i,t} = beta_i * r_market[t] + theta_{i,t} + idio*eps``, ``theta`` an AR(1) idiosyncratic drift.
    Residual momentum (momentum on ``r_i - beta_i*market``) sees the clean ``theta``; total momentum is
    contaminated by beta dispersion and the market's lurches. ``mom_strength = 0`` is the null. Returns
    ``(panel, market, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2009-01-02", periods=n_bars, name="date")
    market = pd.Series(mkt_drift + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")
    betas = np.clip(rng.normal(1.0, 0.3, n_stocks), 0.3, 1.9)
    theta = np.zeros((n_bars, n_stocks))
    innov = mom_strength * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal((n_bars, n_stocks))
    for t in range(1, n_bars):
        theta[t] = phi * theta[t - 1] + innov * nu[t]
    mkt = market.to_numpy()[:, None]
    rets = betas[None, :] * mkt + theta + idio * rng.standard_normal((n_bars, n_stocks))
    panel = pd.DataFrame(rets, index=idx, columns=[f"STK{i:03d}" for i in range(n_stocks)])
    return panel, market, PanelTruth(n_stocks=n_stocks, n_bars=n_bars, mom_strength=mom_strength, phi=phi)


def fetch_panel(start: str = "2010-01-01", min_days: int = 2500, fetch: bool = False,
                cache_dir: str | None = None) -> pd.DataFrame:
    """Current S&P 500 daily-returns panel via :mod:`quantlab.universe`, cache-first."""
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
    rets.index.name = "date"
    return rets
