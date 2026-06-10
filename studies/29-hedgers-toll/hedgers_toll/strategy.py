"""The investable book — long where hedgers are net short, short where they're net long.

Each week, rank the commodities by their (past-only) hedging-pressure signal, go long the top tercile
(commercials most net short → speculators paid the most) and short the bottom, dollar-neutral. This is
the §9.2 hedging-pressure trade. The benchmark is the equal-weight long-only commodity basket: harvesting
the hedging premium is only worth it if the long-short factor adds return the basket doesn't have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hedging import hp_signal

WEEKS_PER_YEAR = 52


def summary(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "skew": float(r.skew()), "n_weeks": int(len(r))}


def _weights(sig, top_frac, long_short):
    cols = sig.columns
    W = pd.DataFrame(0.0, index=sig.index, columns=cols)
    for dt in sig.index:
        row = sig.loc[dt].dropna()
        if len(row) < 4:
            continue
        k = max(1, int(len(row) * top_frac))
        order = row.sort_values()
        W.loc[dt, order.index[-k:]] = 0.5 / k
        if long_short:
            W.loc[dt, order.index[:k]] = -0.5 / k
        else:
            W.loc[dt] = W.loc[dt] / W.loc[dt].abs().sum() if W.loc[dt].abs().sum() > 0 else W.loc[dt]
    return W


def hp_returns(returns: pd.DataFrame, hp: pd.DataFrame, top_frac: float = 0.34, window: int = 156,
               cost_bps: float = 10.0, long_short: bool = True) -> pd.Series:
    """Net weekly return of the hedging-pressure book (long-short top-minus-bottom, or long-only top)."""
    sig = hp_signal(hp, window)
    W = _weights(sig, top_frac, long_short)
    common = W.index.intersection(returns.index)
    W, R = W.loc[common], returns.loc[common]
    gross = (W * R.fillna(0.0)).sum(axis=1)
    cost = (cost_bps * 1e-4) * W.diff().abs().sum(axis=1)
    active = W.abs().sum(axis=1) > 0
    return (gross - cost)[active].rename("hp")


def equal_weight(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1).rename("equal_weight")


def turnover_ann(hp: pd.DataFrame, top_frac=0.34, window=156, long_short=True) -> float:
    W = _weights(hp_signal(hp, window), top_frac, long_short)
    return float(W.diff().abs().sum(axis=1).mean() * WEEKS_PER_YEAR)


def compare(returns: pd.DataFrame, hp: pd.DataFrame, cost_bps: float = 10.0,
            periods_per_year: int = WEEKS_PER_YEAR, **kw) -> dict:
    """The long-short hedging-pressure factor and the long-only-top book vs the equal-weight basket."""
    ls = hp_returns(returns, hp, cost_bps=cost_bps, long_short=True, **kw)
    lo = hp_returns(returns, hp, cost_bps=cost_bps, long_short=False, **kw)
    ew = equal_weight(returns).reindex(ls.index)
    s_ls, s_lo, s_ew = summary(ls, periods_per_year), summary(lo, periods_per_year), summary(ew, periods_per_year)
    return {"long_short": s_ls, "long_only_top": s_lo, "equal_weight": s_ew,
            "turnover_ann": float(turnover_ann(hp, **kw)), "n_weeks": int(s_ls["n_weeks"])}


def cost_sweep(returns, hp, roundtrip_bps=(0, 5, 10, 20, 40), periods_per_year=WEEKS_PER_YEAR, **kw):
    rows = {}
    for c in roundtrip_bps:
        s = summary(hp_returns(returns, hp, cost_bps=c, long_short=True, **kw), periods_per_year)
        rows[c] = {"ls_sharpe": s["sharpe"], "ls_ann": s["ann_return"]}
    out = pd.DataFrame(rows).T; out.index.name = "cost_bps"
    return out
