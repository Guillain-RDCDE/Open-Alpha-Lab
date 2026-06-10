"""Data for the cross-sectional reversion study — an offline synthetic panel with transient
idiosyncratic overshoot, and the real S&P 500 returns panel (the SAME cached tape as Studies 18/24/25).

The desk's offline/cache split:

  * :func:`synthetic_panel` — fully **offline, deterministic**. Each stock carries a common market
    factor plus an **idiosyncratic mean-reverting (Ornstein-Uhlenbeck) deviation**: a name that runs
    ahead of its peers tends to fall back over the next several days, by construction. The cross-section
    is what reverts, so fading the relative leaders earns a market-neutral premium. ``revert_strength``
    sets the effect; ``revert_strength = 0`` is the null (market + pure noise — nothing to fade).
  * :func:`fetch_panel` — a real ``dates × ticker`` daily-return panel for the current S&P 500 via the
    shared :mod:`quantlab.universe` engine, **cache-first** (reads the batched ``panel_503`` download).
    Network only with ``fetch=True``; the import stays lazy so the offline core never needs yfinance.

Two data choices, stated. **Total-return (split/dividend-adjusted) closes** — reversal is a statement
about realised returns. **Current index membership** — the real panel uses *today's* S&P 500, so it
carries **survivorship bias**; the qualitative reversal sign is robust, precise magnitudes are not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
TRADING_DAYS = 252


@dataclass(frozen=True)
class PanelTruth:
    n_stocks: int
    n_bars: int
    revert_strength: float

    @property
    def has_reversion(self) -> bool:
        return self.revert_strength != 0.0


def synthetic_panel(n_stocks: int = 150, n_bars: int = 252 * 16, revert_strength: float = 0.06,
                    mkt_vol: float = 0.009, idio: float = 0.012, seed: int = 33
                    ) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A daily cross-section where **relative winners overshoot and drift back**, by construction.

    For each stock ``i`` an idiosyncratic log-level deviation follows an OU process,
    ``dev_{i,t} = (1-κ)·dev_{i,t-1} + idio·ε``, and the return is
    ``r_{i,t} = β_i·r_market[t] + (dev_{i,t} − dev_{i,t-1})``. Because ``dev`` mean-reverts (κ =
    ``revert_strength``, half-life ≈ ``ln 2 / κ``), a stock that has risen *relative to its level* tends
    to give it back over several days — capturable by a lagged contrarian sort. ``revert_strength = 0``
    removes the pull (a random walk per name): nothing to fade. Returns ``(panel, market, truth)``,
    deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2009-01-02", periods=n_bars, name="date")
    market = pd.Series(0.0003 + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)
    mkt = market.to_numpy()

    dev = np.zeros(n_stocks)
    rets = np.empty((n_bars, n_stocks))
    for t in range(n_bars):
        eps = idio * rng.standard_normal(n_stocks)
        new_dev = (1.0 - revert_strength) * dev + eps     # OU idiosyncratic level
        rets[t] = betas * mkt[t] + (new_dev - dev)        # return = factor + change in level
        dev = new_dev
    cols = [f"STK{i:03d}" for i in range(n_stocks)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    return panel, market, PanelTruth(n_stocks, n_bars, revert_strength)


def fetch_panel(start: str = "2010-01-01", min_days: int = 2500, fetch: bool = False,
                cache_dir: str | None = None) -> pd.DataFrame:
    """Return a ``dates × ticker`` daily-returns panel for the current S&P 500, cache-first.

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
