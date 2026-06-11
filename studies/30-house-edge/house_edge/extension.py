"""Beat-7 worked complement — the markup sweep (by account type), and the drawdown that survives.

Two robustness questions the front-card promises:

  * :func:`financing_sweep` — how the levered book's edge over buy-and-hold changes as the
    broker markup over the bill rises, under BOTH account types (margin: markup on the borrowed
    slice; futures/CFD: markup on the full notional). At a zero markup the two coincide and the
    book ~matches the index; the edge dies *at the retail markup*, and dies fastest in the CFD —
    which is why the mirage is the house's markup, not financing itself.
  * :func:`drawdown_protection` — the part that is *not* a mirage: max-drawdown of the
    vol-targeted book vs buy-and-hold, which holds across account types and markups because
    protection is a risk property, not a return one.

Sharpe everywhere here is the **excess-over-bill** Sharpe (see ``strategy.summary``) so the
financed book and buy-and-hold are compared on the same footing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import buy_and_hold, net_returns
from .strategy import exposure as exposure_fn
from .strategy import summary

MARKUPS = (0.0, 0.01, 0.015, 0.025, 0.05)


def financing_sweep(price: pd.Series, short_rate: pd.Series, target_vol: float = 0.15,
                    markups=MARKUPS, div_yield: float = 0.018) -> pd.DataFrame:
    """CAGR & excess Sharpe of the levered book vs buy-and-hold, as the broker markup rises.

    One row per markup, both account types side by side: ``margin_*`` (markup on the borrowed
    slice only) and ``cfd_*`` (futures-style, markup on the full notional). ``*_edge_cagr`` is
    the CAGR gap to the un-levered total-return hold.
    """
    e = exposure_fn(price, target_vol=target_vol)
    bh = summary(buy_and_hold(price, div_yield=div_yield), rf=short_rate)
    rows = {}
    for m in markups:
        marg = summary(net_returns(e, price, short_rate, mode="margin", markup=m,
                                   div_yield=div_yield), rf=short_rate)
        cfd = summary(net_returns(e, price, short_rate, mode="futures", markup=m,
                                  div_yield=div_yield), rf=short_rate)
        rows[m] = {"margin_cagr": marg["cagr"], "margin_sharpe": marg["sharpe"],
                   "margin_edge_cagr": marg["cagr"] - bh["cagr"],
                   "cfd_cagr": cfd["cagr"], "cfd_sharpe": cfd["sharpe"],
                   "cfd_edge_cagr": cfd["cagr"] - bh["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "markup"
    return out


def drawdown_protection(price: pd.Series, short_rate: pd.Series, target_vol: float = 0.15,
                        markup: float = 0.025, div_yield: float = 0.018) -> dict:
    """Max-drawdown of the vol-targeted book (margin account, retail markup) vs buy-and-hold."""
    e = exposure_fn(price, target_vol=target_vol)
    strat = summary(net_returns(e, price, short_rate, mode="margin", markup=markup,
                                div_yield=div_yield), rf=short_rate)
    bh = summary(buy_and_hold(price, div_yield=div_yield), rf=short_rate)
    return {"strat_max_drawdown": strat["max_drawdown"], "bh_max_drawdown": bh["max_drawdown"],
            "strat_calmar": strat["calmar"], "bh_calmar": bh["calmar"],
            "drawdown_reduction": float(abs(bh["max_drawdown"]) - abs(strat["max_drawdown"]))}
