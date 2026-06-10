"""Data for the alpha-combo study — an offline synthetic panel carrying SEVERAL weak, decorrelated
predictable components, and the real S&P 500 returns panel (the SAME cached tape as Studies 18/24/25/33).

The desk's offline/cache split:

  * :func:`synthetic_panel` — fully **offline, deterministic**. Each stock carries a common market factor
    plus THREE distinct, mutually-decorrelated, individually-thin predictable components, each driven by
    its own independent noise stream:

      1. a slow, persistent relative-performance **drift** (an AR(1) — the cross-sectional *momentum* a
         12-1 sort harvests, as in Study 24 Stampede);
      2. a fast idiosyncratic **overshoot** (an Ornstein-Uhlenbeck deviation — the short-term *reversal*
         a contrarian sort fades, as in Study 33 Slingshot);
      3. a persistent per-name **low-vol tilt** — names with structurally lower idiosyncratic vol earn a
         small positive carry, the *low-volatility* anomaly.

    Each component alone is weak (low standalone Sharpe); because they are driven by independent shocks,
    their captured return streams are near-uncorrelated, so a portfolio of all three is materially
    stronger than any one — the whole point of the study. ``combo_strength`` scales all three at once;
    ``combo_strength = 0`` is the **null** (market + pure noise — nothing for any signal to find).
  * :func:`fetch_panel` — a real ``dates × ticker`` daily-return panel for the current S&P 500 via the
    shared :mod:`quantlab.universe` engine, **cache-first** (reads the batched ``panel_503`` download).
    Network only with ``fetch=True``; the import stays lazy so the offline core never needs yfinance.

Two data choices, stated. **Total-return (split/dividend-adjusted) closes** — every component is a
statement about realised returns. **Current index membership** — the real panel uses *today's* S&P 500,
so it carries **survivorship bias**; the qualitative combo-beats-parts result is robust, precise
magnitudes are not.
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
    """What the synthetic generator baked in, so a test can check the signals recover it."""
    n_stocks: int
    n_bars: int
    combo_strength: float       # scales all three weak components at once; 0 == the null

    @property
    def has_alpha(self) -> bool:
        return self.combo_strength != 0.0


def synthetic_panel(n_stocks: int = 150, n_bars: int = 252 * 16, combo_strength: float = 1.0,
                    mkt_drift: float = 0.0003, mkt_vol: float = 0.009, idio: float = 0.012,
                    phi: float = 0.985, kappa: float = 0.06, seed: int = 38
                    ) -> tuple[pd.DataFrame, pd.Series, PanelTruth]:
    """A daily cross-section carrying three weak, decorrelated predictable components, by construction.

    For each stock ``i`` the return is the sum of a common market factor and three INDEPENDENT
    idiosyncratic pieces, each scaled by ``combo_strength`` (``c`` below):

      * **momentum** — a persistent relative drift ``theta`` (AR(1), persistence ``phi``): a name running
        ahead of the pack tends to keep running ahead, so a 12-1 winners-minus-losers sort earns it;
      * **reversal** — a mean-reverting overshoot ``dev`` (OU, decay ``kappa``): the return picks up the
        *change* in ``dev``, so a name that just jumped relative to peers gives it back, a contrarian
        sort fades it;
      * **low-vol** — each name has a fixed idiosyncratic-vol level; lower-vol names carry a small extra
        positive drift, so a long-low-vol tilt earns it.

    The three are driven by separate RNG draws, so their *captured* return streams are near-uncorrelated.
    Each is deliberately thin (low standalone Sharpe); the combination is much stronger.
    ``combo_strength = 0`` removes all three (market + pure noise — the null). Returns
    ``(panel, market, truth)``, deterministic given ``seed``.
    """
    c = float(combo_strength)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start="2009-01-02", periods=n_bars, name="date")
    market = pd.Series(mkt_drift + mkt_vol * rng.standard_normal(n_bars), index=idx, name="market")
    betas = np.clip(rng.normal(1.0, 0.25, n_stocks), 0.4, 1.8)
    mkt = market.to_numpy()

    # Component 1 — persistent relative drift (cross-sectional momentum). Its own noise stream.
    theta = np.zeros((n_bars, n_stocks))
    mom_amp = c * 0.00085
    innov = mom_amp * np.sqrt(1.0 - phi**2)
    nu = rng.standard_normal((n_bars, n_stocks))
    for t in range(1, n_bars):
        theta[t] = phi * theta[t - 1] + innov * nu[t]

    # Component 3 — per-name structural idiosyncratic-vol level + a low-vol carry. Its own noise stream.
    vol_level = np.clip(rng.normal(idio, idio * 0.4, n_stocks), idio * 0.3, idio * 2.0)
    # lower-vol names get a small positive daily drift (centred so the cross-section nets to ~0)
    lowvol_carry = c * 0.00032 * (idio - vol_level) / idio
    eps_idio = rng.standard_normal((n_bars, n_stocks)) * vol_level[None, :]

    # Component 2 — fast OU overshoot (short-term reversal). Its own noise stream.
    dev = np.zeros(n_stocks)
    rev_amp = c * 0.0085
    rets = np.empty((n_bars, n_stocks))
    for t in range(n_bars):
        shock = rev_amp * rng.standard_normal(n_stocks)
        new_dev = (1.0 - kappa) * dev + shock
        rets[t] = (betas * mkt[t]                 # common factor
                   + theta[t]                     # momentum drift
                   + (new_dev - dev)              # reversal: return = change in overshoot level
                   + lowvol_carry                 # low-vol carry
                   + eps_idio[t])                 # pure idiosyncratic noise
        dev = new_dev

    cols = [f"STK{i:03d}" for i in range(n_stocks)]
    panel = pd.DataFrame(rets, index=idx, columns=cols)
    return panel, market, PanelTruth(n_stocks, n_bars, c)


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
