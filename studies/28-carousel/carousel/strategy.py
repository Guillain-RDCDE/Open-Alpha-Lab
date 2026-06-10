"""The investable book — hold the top-momentum sectors — and the benchmark it must beat.

Sector rotation goes long the top-``k`` sectors by trailing momentum, equal-weight, rebalanced monthly.
The honest benchmark is **holding all sectors equal-weight**: rotation is only worth its turnover and
its concentration (a few bets instead of eleven) if it beats that diversified basket. We also build the
dollar-neutral long-short (top minus bottom) — the pure sector-momentum factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rotation import momentum_score

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "n_days": int(len(r))}


def _weights(panel, score, top_k, rebal, long_short):
    cols = panel.columns; n, m = panel.shape
    W = np.zeros((n, m))
    for p in range(0, n, rebal):
        s = score.iloc[p].dropna()
        if len(s) < (2 * top_k if long_short else top_k):
            continue
        order = s.sort_values()
        end = min(p + rebal, n)
        W[p:end, cols.get_indexer(order.index[-top_k:])] = 1.0 / top_k
        if long_short:
            W[p:end, cols.get_indexer(order.index[:top_k])] = -1.0 / top_k
    return pd.DataFrame(W, index=panel.index, columns=cols)


def rotation_returns(panel: pd.DataFrame, top_k: int = 3, lookback: int = 126, skip: int = 21,
                     rebal: int = 21, cost_bps: float = 3.0, long_short: bool = False) -> pd.Series:
    """Net daily return of the top-``k`` sector-rotation book (long-only, or long-short top-minus-bottom)."""
    score = momentum_score(panel, lookback, skip)
    W = _weights(panel, score, top_k, rebal, long_short)
    gross = (W.shift(1) * panel.fillna(0.0)).sum(axis=1)
    cost = (cost_bps * 1e-4) * W.diff().abs().sum(axis=1)
    active = W.abs().sum(axis=1) > 0
    return (gross - cost)[active].rename("rotation")


def equal_weight(panel: pd.DataFrame) -> pd.Series:
    return panel.mean(axis=1).rename("equal_weight")


def turnover_ann(panel: pd.DataFrame, top_k=3, lookback=126, skip=21, rebal=21, long_short=False) -> float:
    W = _weights(panel, momentum_score(panel, lookback, skip), top_k, rebal, long_short)
    return float(W.diff().abs().sum(axis=1).mean() * TRADING_DAYS_PER_YEAR)


def compare(panel: pd.DataFrame, top_k: int = 3, cost_bps: float = 3.0,
            periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Long-only rotation and the long-short factor vs the equal-weight sector basket."""
    ro = rotation_returns(panel, top_k=top_k, cost_bps=cost_bps, long_short=False, **kw)
    ls = rotation_returns(panel, top_k=top_k, cost_bps=cost_bps, long_short=True, **kw)
    ew = equal_weight(panel).reindex(ro.index)
    s_ro, s_ls, s_ew = summary(ro, periods_per_year), summary(ls, periods_per_year), summary(ew, periods_per_year)
    return {"rotation": s_ro, "long_short": s_ls, "equal_weight": s_ew,
            "rotation_minus_ew_sharpe": float(s_ro["sharpe"] - s_ew["sharpe"]),
            "turnover_ann": float(turnover_ann(panel, top_k=top_k, **kw)), "n_days": int(s_ro["n_days"])}


def cost_sweep(panel, top_k=3, roundtrip_bps=(0, 3, 5, 10, 20), periods_per_year=TRADING_DAYS_PER_YEAR, **kw):
    rows = {}
    for c in roundtrip_bps:
        cmp = compare(panel, top_k=top_k, cost_bps=c, periods_per_year=periods_per_year, **kw)
        rows[c] = {"rotation_sharpe": cmp["rotation"]["sharpe"], "rotation_minus_ew": cmp["rotation_minus_ew_sharpe"]}
    out = pd.DataFrame(rows).T; out.index.name = "cost_bps"
    return out
