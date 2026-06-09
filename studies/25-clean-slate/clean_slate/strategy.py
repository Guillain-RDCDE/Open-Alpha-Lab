"""The investable books — residual-momentum WML, and the total-momentum WML it's meant to improve on.

We build the same long-winners/short-losers WML factor two ways: ranked on the **residual** 12-1 score
(the study's strategy) and on the **total**-return score ([Study 24](../../24-stampede/)'s factor). The
two books trade the same names with the same costs; the comparison — same premium, gentler ride — is the
whole point. The benchmark is the equal-weight universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import residual_returns, momentum_score

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
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "skew": float(r.skew()), "n_days": int(len(r))}


def _wml(panel, score, rebal, decile_frac, cost_bps):
    cols = panel.columns; n, m = panel.shape
    w = np.zeros((n, m))
    for p in range(0, n, rebal):
        s = score.iloc[p].dropna()
        if len(s) < max(10, int(2 / decile_frac)):
            continue
        k = max(1, int(len(s) * decile_frac))
        order = s.sort_values()
        end = min(p + rebal, n)
        w[p:end, cols.get_indexer(order.index[-k:])] = 1.0 / k
        w[p:end, cols.get_indexer(order.index[:k])] = -1.0 / k
    wdf = pd.DataFrame(w, index=panel.index, columns=cols)
    gross = (wdf.shift(1) * panel.fillna(0.0)).sum(axis=1)
    cost = (cost_bps * 1e-4) * wdf.diff().abs().sum(axis=1)
    active = wdf.abs().sum(axis=1) > 0
    return (gross - cost)[active].rename("wml"), float(wdf.diff().abs().sum(axis=1).mean() * TRADING_DAYS_PER_YEAR)


def wml_returns(panel: pd.DataFrame, market: pd.Series | None = None, residual: bool = True,
                lookback: int = 252, skip: int = 21, rebal: int = 21, decile_frac: float = 0.1,
                cost_bps: float = 5.0) -> pd.Series:
    """The WML net stream, ranked on residual (``residual=True``) or total 12-1 momentum."""
    rr = residual_returns(panel, market) if residual else panel
    score = momentum_score(rr, lookback, skip)
    return _wml(panel, score, rebal, decile_frac, cost_bps)[0]


def turnover_ann(panel: pd.DataFrame, market=None, residual=True, **kw) -> float:
    rr = residual_returns(panel, market) if residual else panel
    score = momentum_score(rr, kw.get("lookback", 252), kw.get("skip", 21))
    return _wml(panel, score, kw.get("rebal", 21), kw.get("decile_frac", 0.1), kw.get("cost_bps", 5.0))[1]


def compare(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
            periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Residual WML vs total WML vs the equal-weight market — the scoreboard."""
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    tot = wml_returns(panel, market, residual=False, cost_bps=cost_bps, **kw)
    mkt = panel.mean(axis=1).reindex(res.index)
    return {"residual": summary(res, periods_per_year), "total": summary(tot, periods_per_year),
            "market": summary(mkt, periods_per_year),
            "turnover_ann": float(turnover_ann(panel, market, residual=True, cost_bps=cost_bps, **kw)),
            "n_days": int(summary(res, periods_per_year)["n_days"])}


def cost_sweep(panel, market=None, roundtrip_bps=(0, 5, 10, 20, 40), periods_per_year=TRADING_DAYS_PER_YEAR, **kw):
    rows = {}
    for c in roundtrip_bps:
        s = summary(wml_returns(panel, market, residual=True, cost_bps=c, **kw), periods_per_year)
        rows[c] = {"residual_sharpe": s["sharpe"], "residual_ann": s["ann_return"]}
    out = pd.DataFrame(rows).T; out.index.name = "cost_bps"
    return out
