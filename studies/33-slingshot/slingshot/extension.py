"""Beat-7 worked complement — "has the reversal premium decayed, and can a slower book bank it?"

Two questions the base run leaves open:
  * :func:`subperiod_sharpe` — short-term reversal in equities was hugely profitable pre-2000 and is
    widely reported to have **decayed** as statistical-arbitrage desks and then HFT competed it away
    (Khandani & Lo 2007/2011). We split the modern sample and look for that fade.
  * :func:`holding_period_sweep` — the same cost rescue as Rip-Tide: rebalance every ``h`` days to cut
    turnover. Does any holding period push the *net* Sharpe above zero at realistic equity costs?

Plus :func:`lookback_sweep` — is the effect a knife-edge 1-day bounce or does it survive to a week?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import reversal_signal, summary


def _book_from_weights(w: pd.DataFrame, returns: pd.DataFrame, cost_bps: float) -> pd.Series:
    gross = (w.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("reversion")


def lookback_sweep(returns: pd.DataFrame, lookbacks=(1, 3, 5, 10, 21), cost_bps: float = 5.0
                   ) -> pd.DataFrame:
    """Net Sharpe / CAGR of the book for several reversal lookbacks (1 day → 1 month)."""
    from .strategy import book_returns
    rows = {}
    for lb in lookbacks:
        s = summary(book_returns(returns, lookback=lb, cost_bps=cost_bps))
        rows[lb] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "lookback_days"
    return out


def holding_period_sweep(returns: pd.DataFrame, holds=(1, 2, 5, 10, 21), lookback: int = 5,
                         cost_bps: float = 5.0) -> pd.DataFrame:
    """Hold each day's weights ``h`` days (rebalance every ``h``-th day): gross Sharpe, net Sharpe and
    turnover. Slowing down cuts cost but stales the signal — does the net frontier clear zero?"""
    base = reversal_signal(returns, lookback=lookback)
    rows = {}
    for h in holds:
        if h <= 1:
            w = base
        else:
            mask = np.zeros(len(base), dtype=bool)
            mask[::h] = True
            w = base.where(pd.Series(mask, index=base.index), other=np.nan).ffill().fillna(0.0)
        gross = summary(_book_from_weights(w, returns, 0.0))["sharpe"]
        net = summary(_book_from_weights(w, returns, cost_bps))["sharpe"]
        tpd = float(w.diff().abs().sum(axis=1).mean())
        rows[h] = {"gross_sharpe": gross, "net_sharpe": net, "turnover_per_day": tpd}
    out = pd.DataFrame(rows).T
    out.index.name = "hold_days"
    return out


def subperiod_sharpe(returns: pd.DataFrame, n_splits: int = 3, lookback: int = 5, cost_bps: float = 5.0
                     ) -> pd.DataFrame:
    """Net Sharpe of the book across equal-length sub-periods — to expose the well-known decay."""
    from .strategy import book_returns
    book = book_returns(returns, lookback=lookback, cost_bps=cost_bps).dropna()
    bounds = np.linspace(0, len(book), n_splits + 1).astype(int)
    rows = {}
    for i in range(n_splits):
        seg = book.iloc[bounds[i]:bounds[i + 1]]
        s = summary(seg)
        rows[f"{seg.index[0].date()} → {seg.index[-1].date()}"] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "sub_period"
    return out
