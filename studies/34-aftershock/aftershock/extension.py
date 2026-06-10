"""Beat-7 worked complement — the **drift-decay curve** and the holding-period sweep.

Two questions the base run leaves open:

  * :func:`drift_decay_curve` — the canonical PEAD picture (Bernard-Thomas 1989, Fig. 1): line up every
    earnings event at τ=0 and average the **surprise-signed cumulative abnormal return** by trading-day-
    since-announcement. If under-reaction is real, the curve rises steadily for weeks and then flattens —
    that *shape* is the whole anomaly, and it is what tells you how long you must hold.
  * :func:`holding_period_sweep` — does the drift persist long enough, and pay enough, to clear costs?
    Rebalance/hold for ``h`` days and read gross Sharpe, net Sharpe and turnover. Too short and you miss
    the drift; too long and the signal has decayed to noise while you keep paying to hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, summary


def drift_decay_curve(panel: pd.DataFrame, events: pd.DataFrame, window: int = 70) -> pd.Series:
    """Average **surprise-signed** cumulative abnormal return by trading-day since the announcement.

    For each event, take the sign of its surprise and accumulate ``sign(s) · r`` over the ``window``
    trading days starting on the event day, then average across all events. A rising-then-flattening curve
    is the PEAD under-reaction signature; a flat curve (the null) means surprises carry no information.
    Returns a Series indexed by ``days_since_event`` (0 … window-1) of mean cumulative signed return.
    """
    idx = panel.index
    mat = panel.to_numpy()
    col = {c: k for k, c in enumerate(panel.columns)}
    acc = np.zeros(window)
    n = 0
    for _, row in events.iterrows():
        tk = row["ticker"]
        if tk not in col:
            continue
        loc = idx.searchsorted(pd.Timestamp(row["date"]), side="left")
        if loc >= len(idx):
            continue
        hi = min(loc + window, len(idx))
        seg = mat[loc:hi, col[tk]] * np.sign(row["surprise"])
        if len(seg) < window:
            continue
        acc += np.cumsum(seg)
        n += 1
    out = pd.Series(acc / max(n, 1), index=pd.RangeIndex(window, name="days_since_event"),
                    name="mean_signed_car")
    return out


def holding_period_sweep(panel: pd.DataFrame, events: pd.DataFrame, holds=(5, 20, 40, 60, 90),
                         cost_bps: float = 5.0) -> pd.DataFrame:
    """Gross Sharpe, net Sharpe and turnover of the PEAD book for several holding horizons. Short holds
    capture only the first, steepest part of the drift; long holds keep paying to hold a signal that has
    decayed. The sweet spot is where the drift still pays more than the roll costs."""
    from .strategy import turnover as _turnover
    rows = {}
    for h in holds:
        gross = summary(book_returns(panel, events, holding_days=h, cost_bps=0.0))["sharpe"]
        net = summary(book_returns(panel, events, holding_days=h, cost_bps=cost_bps))["sharpe"]
        tpd = _turnover(panel, events, holding_days=h)
        rows[h] = {"gross_sharpe": gross, "net_sharpe": net, "turnover_per_day": tpd}
    out = pd.DataFrame(rows).T
    out.index.name = "hold_days"
    return out
