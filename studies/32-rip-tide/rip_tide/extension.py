"""Beat-7 worked complement — "can you save reversion by slowing it down?"

The base book trades every day and dies on costs. The obvious rescue is to **rebalance less often**:
hold each contrarian position for ``h`` days instead of one. That cuts turnover (and thus cost) roughly
in proportion to ``h`` — but it also lets the position drift away from the freshly-computed signal, so
the gross edge fades too. :func:`holding_period_sweep` traces that trade-off and asks whether *any*
holding period clears realistic costs.

Two supporting checks:
  * :func:`horizon_sweep` — does the reversal pay only at a magic lookback, or across short horizons?
  * :func:`subperiod_sharpe` — has short-term reversion decayed as markets got more efficient?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import TRADING_DAYS, positions, summary


def _book_from_positions(pos: pd.DataFrame, returns: pd.DataFrame, cost_bps: float) -> pd.Series:
    gross = (pos.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * pos.diff().abs().sum(axis=1)
    return (gross - cost).rename("reversion")


def horizon_sweep(returns: pd.DataFrame, lookbacks_list=((1,), (3,), (5,), (10,), (1, 3, 5)),
                  cost_bps: float = 2.0, **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the book for several short-lookback choices (single horizons + the blend)."""
    from .strategy import book_returns
    rows = {}
    for lb in lookbacks_list:
        s = summary(book_returns(returns, lookbacks=lb, cost_bps=cost_bps, **kw))
        rows["+".join(str(x) for x in lb)] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "lookback_days"
    return out


def holding_period_sweep(returns: pd.DataFrame, holds=(1, 2, 5, 10, 21), cost_bps: float = 2.0,
                         **kw) -> pd.DataFrame:
    """The beat-7 rescue attempt: hold each position ``h`` days (rebalance every ``h``-th day) and
    report gross Sharpe, net Sharpe and average turnover. Slowing down cuts cost but dilutes the
    signal — this shows whether the net frontier ever pokes above zero."""
    base = positions(returns, **kw)
    rows = {}
    for h in holds:
        if h <= 1:
            pos = base
        else:
            mask = np.zeros(len(base), dtype=bool)
            mask[::h] = True                       # rebalance dates
            pos = base.where(pd.Series(mask, index=base.index), other=np.nan).ffill().fillna(0.0)
        gross = summary(_book_from_positions(pos, returns, 0.0))["sharpe"]
        net = summary(_book_from_positions(pos, returns, cost_bps))["sharpe"]
        tpd = float(pos.diff().abs().sum(axis=1).mean())
        rows[h] = {"gross_sharpe": gross, "net_sharpe": net, "turnover_per_day": tpd}
    out = pd.DataFrame(rows).T
    out.index.name = "hold_days"
    return out


def subperiod_sharpe(returns: pd.DataFrame, n_splits: int = 3, cost_bps: float = 2.0, **kw) -> pd.DataFrame:
    """Net Sharpe of the book across equal-length sub-periods — to expose decay as markets got faster."""
    from .strategy import book_returns
    book = book_returns(returns, cost_bps=cost_bps, **kw).dropna()
    bounds = np.linspace(0, len(book), n_splits + 1).astype(int)
    rows = {}
    for i in range(n_splits):
        seg = book.iloc[bounds[i]:bounds[i + 1]]
        s = summary(seg)
        rows[f"{seg.index[0].date()} → {seg.index[-1].date()}"] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "sub_period"
    return out
