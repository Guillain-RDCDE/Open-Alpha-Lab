"""Beat-7 worked complement — the financing-rate sweep, and the drawdown that survives.

Two robustness questions the front-card promises:

  * :func:`financing_sweep` — how the levered strategy's honest return advantage over buy-and-hold
    changes as the broker markup rises. The "leverage edge" is a *decreasing* function of the
    financing cost — at realistic markups it's gone, which is why it's a MIRAGE, not a tuning miss.
  * :func:`drawdown_protection` — the part that is *not* a mirage: max-drawdown of the vol-targeted
    book vs buy-and-hold, which holds across cost models because protection is a risk property, not
    a return one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import buy_and_hold, net_returns
from .strategy import exposure as exposure_fn
from .strategy import summary


def financing_sweep(price: pd.Series, short_rate: pd.Series, target_vol: float = 0.15,
                    markups=(0.0, 0.01, 0.025, 0.05), div_yield: float = 0.018) -> pd.DataFrame:
    """Honest CAGR & Sharpe of the levered book vs buy-and-hold, as the financing markup rises."""
    e = exposure_fn(price, target_vol=target_vol)
    bh = summary(buy_and_hold(price, div_yield=div_yield))
    rows = {}
    for m in markups:
        s = summary(net_returns(e, price, short_rate, mode="honest", markup=m, div_yield=div_yield))
        rows[m] = {"strat_cagr": s["cagr"], "strat_sharpe": s["sharpe"],
                   "edge_vs_bh_cagr": s["cagr"] - bh["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "markup"
    return out


def drawdown_protection(price: pd.Series, short_rate: pd.Series, target_vol: float = 0.15,
                        div_yield: float = 0.018) -> dict:
    """Max-drawdown of the vol-targeted book (honest costs) vs buy-and-hold — the surviving edge."""
    e = exposure_fn(price, target_vol=target_vol)
    strat = summary(net_returns(e, price, short_rate, mode="honest", div_yield=div_yield))
    bh = summary(buy_and_hold(price, div_yield=div_yield))
    return {"strat_max_drawdown": strat["max_drawdown"], "bh_max_drawdown": bh["max_drawdown"],
            "strat_calmar": strat["calmar"], "bh_calmar": bh["calmar"],
            "drawdown_reduction": float(abs(bh["max_drawdown"]) - abs(strat["max_drawdown"]))}
