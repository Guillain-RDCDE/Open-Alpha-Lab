"""Costs and benchmarks — why the trend book's tradability question is friendlier than most.

Time-series momentum on liquid futures has an unusually honest cost story:
  * **No full-notional financing.** A future embeds the cost of carry in its price; you post a small
    margin and *earn* interest on the rest of your cash. There is no CFD-style daily financing drag on
    the whole notional (that was Study 30's killer) — so leverage here is genuinely cheap.
  * **Tiny transaction costs.** The basket is the most liquid stuff on the planet (ES, ZN, CL, GC, 6E);
    round-trip costs are ~1-3 bp of notional. The book trades slowly (monthly-ish signal).

So `book_returns` (in :mod:`trade_winds.strategy`) already nets transaction cost; this module adds the
**cost sweep** (does the edge survive higher costs?) and the **passive benchmarks** the trend book must
beat to be worth running: a 60/40 stock/bond hold, and the equal-risk long-only basket (always-long the
same markets, same vol-scaling) — to prove the *trend timing* adds something the diversification alone
doesn't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import TRADING_DAYS, book_returns, positions, summary


def cost_sweep(returns: pd.DataFrame, roundtrip_bps=(0, 1, 2, 5, 10), **kw) -> pd.DataFrame:
    """Net Sharpe / CAGR of the TSMOM book as the per-trade cost rises — does the edge survive?"""
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(returns, cost_bps=c, **kw))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def long_only_basket(returns: pd.DataFrame, vol_window: int = 63, target_vol: float = 0.10) -> pd.Series:
    """Always-long, equal-risk, vol-scaled basket of the SAME markets — the diversification-only
    benchmark. If the trend book doesn't beat this, the *timing* adds nothing over just owning the
    diversified basket."""
    realised = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    per_mkt = target_vol / np.sqrt(max(1, returns.shape[1]))
    w = (per_mkt / realised).clip(upper=3.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (w.shift(1) * returns).sum(axis=1).rename("long_only_basket")


def crisis_alpha(returns: pd.DataFrame, equity_cols, **kw) -> dict:
    """Does the trend book pay when equities fall? Returns the book's return in the worst equity
    months vs the rest, and its correlation to the equity sleeve — the 'crisis alpha' signature."""
    book = book_returns(returns, **kw)
    eq = returns[[c for c in equity_cols if c in returns.columns]].mean(axis=1).reindex(book.index)
    eq_m = (1 + eq).resample("ME").prod() - 1
    bk_m = (1 + book).resample("ME").prod() - 1
    worst = eq_m.nsmallest(max(3, len(eq_m) // 10)).index          # worst ~decile equity months
    return {"book_in_equity_crises_ann": float(bk_m.loc[worst].mean() * 12),
            "book_rest_ann": float(bk_m.loc[~bk_m.index.isin(worst)].mean() * 12),
            "corr_to_equities": float(book.corr(eq)),
            "n_crisis_months": int(len(worst))}
