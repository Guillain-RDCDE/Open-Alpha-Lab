"""Beat-7 worked complement — the lookback sweep and the decay-through-time check.

Two robustness questions:
  * :func:`lookback_sweep` — does the trend edge depend on a magic lookback, or pay across the board?
    (TSMOM should be positive from ~1 month to ~12 months — a real premium, not a tuned parameter.)
  * :func:`subperiod_sharpe` — the honest one: trend-following had a well-documented *drought* in the
    2010s. We split the sample and show where it paid and where it didn't, so the verdict is earned,
    not cherry-picked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, summary


def lookback_sweep(returns: pd.DataFrame, lookbacks_list=((21,), (63,), (126,), (252,), (21, 63, 252)),
                   cost_bps: float = 2.0, **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the book for several lookback choices (single horizons + the blend)."""
    rows = {}
    for lb in lookbacks_list:
        s = summary(book_returns(returns, lookbacks=lb, cost_bps=cost_bps, **kw))
        rows["+".join(str(x) for x in lb)] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "lookback_days"
    return out


def breadth_sweep(returns: pd.DataFrame, ks=None, n_draws: int = 40, cost_bps: float = 2.0,
                  seed: int = 31, **kw) -> pd.DataFrame:
    """The beat-7 breadth complement: mean book Sharpe as a function of the NUMBER of markets.

    For each ``k`` we average the net-of-cost TSMOM Sharpe over ``n_draws`` random size-``k`` subsets of
    the universe (the full universe for the largest ``k``). Trend-following's Sharpe is a *breadth*
    story — this shows whether the curve is still rising (i.e. more markets would help) or has plateaued.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = returns.shape[1]
    if ks is None:
        ks = sorted({1, 2, 4, 8, 12, min(n, 16), n})
    rows = {}
    for k in ks:
        if k > n:
            continue
        sharpes = []
        draws = 1 if k == n else n_draws
        for _ in range(draws):
            cols = list(returns.columns) if k == n else list(rng.choice(returns.columns, size=k, replace=False))
            sharpes.append(summary(book_returns(returns[cols], cost_bps=cost_bps, **kw))["sharpe"])
        rows[k] = {"mean_sharpe": float(np.nanmean(sharpes)),
                   "sharpe_p25": float(np.nanpercentile(sharpes, 25)),
                   "sharpe_p75": float(np.nanpercentile(sharpes, 75))}
    out = pd.DataFrame(rows).T
    out.index.name = "n_markets"
    return out


def subperiod_sharpe(returns: pd.DataFrame, n_splits: int = 3, cost_bps: float = 2.0, **kw) -> pd.DataFrame:
    """Net Sharpe of the book across equal-length sub-periods — to expose decay (the 2010s drought)."""
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
