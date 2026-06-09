"""Data access for the cross-sectional momentum study — the panel and where it comes from.

Cross-sectional momentum lives in the *relative* race: rank stocks by their past year's return and bet
the leaders keep leading. So the tape is a panel of daily returns for many names, and the data layer
keeps the desk's offline/cache split:

    * :func:`synthetic_panel` — fully **offline**. A panel where each stock carries a slow, *persistent*
      relative-performance drift ``theta_{i,t}`` (an AR(1)) on top of a common market factor and idio
      noise. Because the drift persists, a stock that has out-run the pack tends to keep out-running it —
      the momentum the strategy harvests. ``mom_strength`` sets the drift's amplitude; pass
      ``mom_strength = 0`` for the **null** — only the market and noise, so past winners and losers are
      indistinguishable going forward and a momentum sort earns nothing. Deterministic given ``seed``.
    * :func:`fetch_panel` — a real cross-section (current S&P 500 constituents) via the shared
      :mod:`quantlab.universe` engine, reduced to a daily-returns panel. **Cache-only** unless
      ``fetch=True``; the network import stays lazy, so the offline core never imports ``yfinance``.

Two data choices, named up front. **Total-return (split/dividend-adjusted) closes** — momentum is a
statement about total realized returns. **Current index membership** — the real panel uses *today's*
S&P 500, so it carries **survivorship bias** (delisted names excluded); the qualitative winners-minus-
losers ranking is robust, precise magnitudes are not, and we say so wherever a real number appears.
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
    """What the synthetic generator baked in, so a test can check the diagnostics recover it."""
    n_stocks: int
    n_bars: int
    mom_strength: float       # amplitude of the persistent relative drift; 0 == no-momentum null
    phi: float                # AR(1) persistence of the relative drift
    mkt_vol: float

    @property
    def has_momentum(self) -> bool:
        return self.mom_strength != 0.0


def synthetic_panel(
    n_stocks: int = 150,
    n_bars: int = 4032,
    mom_strength: float = 0.0015,
    phi: float = 0.985,
    mkt_drift: float = 0.0003,
    mkt_vol: float = 0.009,
    idio: float = 0.012,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A daily cross-section where **past winners keep winning**, by construction.

    For each stock ``i``::

        r_market[t] = mkt_drift + mkt_vol * z[t]                       (one shared factor)
        theta_{i,t} = phi * theta_{i,t-1} + sqrt(1-phi^2)*mom_strength*nu_{i,t}   (persistent rel. drift)
        r_{i,t}     = beta_i * r_market[t] + theta_{i,t} + idio * eps_{i,t}

    The relative drift ``theta`` persists (``phi`` near 1), so a stock's trailing return is a good read
    on its *current* drift, which carries into the next period — the time-series root of cross-sectional
    momentum. With ``mom_strength = 0`` there is no persistent relative drift: past winners and losers
    are just noisy, and a momentum sort earns nothing (the null). Returns ``(panel, market, truth)``.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2009-01-02", periods=n_bars, name="date")
    market = pd.Series(mkt_drift + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)

    theta = np.zeros((n_bars, n_stocks))
    innov = mom_strength * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal((n_bars, n_stocks))
    for t in range(1, n_bars):
        theta[t] = phi * theta[t - 1] + innov * nu[t]
    mkt = market.to_numpy()[:, None]
    eps = rng.standard_normal((n_bars, n_stocks))
    rets = betas[None, :] * mkt + theta + idio * eps

    cols = [f"STK{i:03d}" for i in range(n_stocks)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    truth = PanelTruth(n_stocks=n_stocks, n_bars=n_bars, mom_strength=mom_strength, phi=phi, mkt_vol=mkt_vol)
    return panel, market, truth


def fetch_panel(start: str = "2010-01-01", min_days: int = 2500, fetch: bool = False,
                cache_dir: str | None = None) -> pd.DataFrame:
    """Return a ``dates x ticker`` daily-returns panel for the current S&P 500, cache-first.

    Leans on :mod:`quantlab.universe`: reads the cached batched OHLC download, keeps names with at least
    ``min_days`` of (split/dividend-adjusted) closes, and differences to returns. **Cache-only by
    default**; ``fetch=True`` lets the engine hit Yahoo once. The network import stays lazy.
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
    closes = {}
    for tk, ohlc in panel.items():
        c = ohlc["Close"].dropna()
        if len(c) >= min_days:
            closes[tk] = c
    if not closes:
        return pd.DataFrame()
    rets = pd.DataFrame(closes).sort_index().pct_change().dropna(how="all")
    rets.index.name = "date"
    return rets
