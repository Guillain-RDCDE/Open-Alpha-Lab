"""The book — combine several weak, decorrelated component signals into one portfolio.

The thesis of the whole study (Kakushadze-Serur §3.20, "alpha combo"): no single anomaly is impressive
alone, but a portfolio of several *weak, decorrelated* signals has a materially better Sharpe than any
component. That is the Fundamental Law of Active Management at work — Sharpe ≈ IC · √breadth — and the
desk's recurring lesson that **the edge is diversification, not prediction.**

The combiner takes the component weight-streams (each already dollar-neutral, gross-1, lagged) and blends
them into one book, under two schemes:

  * **equal-weight** — average the components' weights (each gets 1/N), the naive blend;
  * **risk-parity** (inverse-vol) — weight each component by the inverse of its own standalone book
    volatility, so a noisier signal doesn't dominate; the volatilities are estimated on a burn-in slice
    to keep the blend causal.

The blended weight-stream is then itself renormalised to gross 1 each day, so the combo and every
component are compared on the same unit of risk budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def book_returns(weights: pd.DataFrame, panel: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """Net daily return of a (lagged, gross-1) weight-stream: weights applied to that day's returns minus
    turnover cost (``cost_bps`` per unit of |Δw| traded). The weights are already lagged inside the signal
    functions, so they multiply the *current* day's return directly."""
    aligned = weights.reindex_like(panel).fillna(0.0)
    gross = (aligned * panel).sum(axis=1)
    cost = (cost_bps * 1e-4) * aligned.diff().abs().sum(axis=1)
    return (gross - cost).rename("book")


def _renorm_gross1(w: pd.DataFrame) -> pd.DataFrame:
    """Renormalise a combined weight-stream back to gross exposure 1 each day (rows already ~neutral)."""
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def combine(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, scheme: str = "equal",
            burn_in: int = 252) -> pd.DataFrame:
    """Blend the component weight-streams into one gross-1 book.

    ``scheme='equal'`` averages the components (each 1/N). ``scheme='risk_parity'`` weights each by the
    inverse of its standalone book volatility, estimated on the first ``burn_in`` days only (so the
    allocation uses no future information), then held fixed. Returns the combined, renormalised
    weight-stream (still dollar-neutral, gross 1, and causal — it is a fixed linear blend of causal
    inputs)."""
    names = list(signals)
    if scheme == "equal":
        coef = {nm: 1.0 for nm in names}
    elif scheme == "risk_parity":
        coef = {}
        for nm in names:
            r = book_returns(signals[nm], panel, cost_bps=0.0).iloc[:burn_in]
            sd = r.std(ddof=1)
            coef[nm] = 1.0 / sd if sd and np.isfinite(sd) and sd > 0 else 0.0
    else:
        raise ValueError(f"unknown scheme {scheme!r} (expected 'equal' or 'risk_parity')")

    total = sum(coef.values()) or 1.0
    blended = None
    for nm in names:
        contrib = signals[nm] * (coef[nm] / total)
        blended = contrib if blended is None else blended.add(contrib, fill_value=0.0)
    return _renorm_gross1(blended)


def turnover(weights: pd.DataFrame) -> float:
    """Average daily one-way turnover (Σ|Δw|) of a weight-stream."""
    return float(weights.diff().abs().sum(axis=1).mean())


def avg_pairwise_corr(signals: dict[str, pd.DataFrame], panel: pd.DataFrame) -> float:
    """Average pairwise correlation of the components' standalone (gross, cost-free) return streams — the
    decorrelation that makes the combo work. Near zero is the goal."""
    rets = pd.DataFrame({nm: book_returns(w, panel, cost_bps=0.0) for nm, w in signals.items()}).dropna()
    if rets.shape[1] < 2:
        return np.nan
    cm = rets.corr().to_numpy()
    iu = np.triu_indices_from(cm, k=1)
    return float(np.nanmean(cm[iu]))


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_days": int(len(r))}
