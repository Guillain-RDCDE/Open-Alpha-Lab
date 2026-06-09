"""The investable book — long the high-rate currencies, short the low-rate ones, rebalanced monthly.

The carry portfolio is dollar-neutral: each month, rank the currencies by short rate and go long the
top, short the bottom (weighted by the rate gap from the cross-sectional average). Rates move slowly, so
turnover is *low* and cost is not the threat — the threat is the **crash**: carry is short the world's
funding currencies and long its risky high-yielders, so in a global risk-off it loses on every leg at
once. The benchmark is being flat (zero) — carry is an absolute-return, self-financed trade.

Everything is monthly; we annualise by ``×12`` (mean) and ``×√12`` (vol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def summary(returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "ann_return": float(mean * periods_per_year), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "cagr": float(cagr), "max_drawdown": float(dd.min()), "skew": float(r.skew()), "n_months": int(len(r))}


def carry_weights(rates: pd.DataFrame, decile_frac: float = 0.34) -> pd.DataFrame:
    """Dollar-neutral monthly weights: long the highest-rate, short the lowest-rate currencies.

    Each month the top and bottom ``decile_frac`` of currencies by rate are equal-weighted long and
    short. Weights are lagged one month (set on the rate known at month-end, earning next month's return).
    """
    cols = rates.columns
    W = np.zeros((len(rates), len(cols)))
    rv = rates.to_numpy()
    for t in range(len(rates)):
        row = rv[t]
        valid = np.where(~np.isnan(row))[0]
        if len(valid) < 3:
            continue
        k = max(1, int(len(valid) * decile_frac))
        order = valid[np.argsort(row[valid])]
        W[t, order[-k:]] = 0.5 / k
        W[t, order[:k]] = -0.5 / k
    return pd.DataFrame(W, index=rates.index, columns=cols).shift(1)


def carry_returns(xret: pd.DataFrame, rates: pd.DataFrame, decile_frac: float = 0.34,
                  cost_bps: float = 10.0) -> pd.Series:
    """The carry portfolio's monthly return, net of ``cost_bps`` per unit of weight traded."""
    w = carry_weights(rates, decile_frac).reindex(xret.index)
    gross = (w * xret).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).dropna().rename("carry")


def turnover_ann(rates: pd.DataFrame, decile_frac: float = 0.34) -> float:
    w = carry_weights(rates, decile_frac)
    return float(w.diff().abs().sum(axis=1).mean() * MONTHS_PER_YEAR)


def compare(xret: pd.DataFrame, rates: pd.DataFrame, cost_bps: float = 10.0,
            periods_per_year: int = MONTHS_PER_YEAR, **kw) -> dict:
    """Headline stats for the carry book, plus its (low) turnover."""
    r = carry_returns(xret, rates, cost_bps=cost_bps, **kw)
    s = summary(r, periods_per_year)
    return {**s, "turnover_ann": float(turnover_ann(rates, kw.get("decile_frac", 0.34)))}


def cost_sweep(xret, rates, roundtrip_bps=(0, 10, 25, 50, 100), periods_per_year=MONTHS_PER_YEAR, **kw):
    rows = {}
    for c in roundtrip_bps:
        s = summary(carry_returns(xret, rates, cost_bps=c, **kw), periods_per_year)
        rows[c] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T; out.index.name = "cost_bps"
    return out
