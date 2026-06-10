"""The book — a cross-sectional commodity **carry** book: long the most-backwardated, short the most-contangoed.

Each week:
  1. **Signal** — each commodity's roll yield (term-structure carry). Higher roll yield = more
     backwardated = expected to out-return. We rank the cross-section by roll yield.
  2. **Dollar-neutral weights** — the cross-sectionally demeaned roll yield, normalised so the long and
     short legs net to zero and gross exposure sums to one. Long the high-carry names, short the low-carry
     ones. Lagged one week so the signal is causal/tradable.
  3. **Weekly rebalance** — roll yield is a slow signal, so the book turns over modestly; that low turnover
     is what makes commodity carry comparatively *cheap* to run (the contrast with Study 33's daily churn).

This is the commodity sibling of Study 27 (Steamroller, FX carry): both rank a cross-section by a carry
signal (rate differential there, roll yield here), go long-high / short-low, and harvest a documented
premium that is real but crash-prone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def carry_signal(roll_yield: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral carry weights: the cross-sectionally demeaned roll yield, normalised so gross
    exposure is 1 each week, lagged one week. Long the most-backwardated (high roll yield), short the
    most-contangoed (low roll yield)."""
    x = roll_yield.sub(roll_yield.mean(axis=1), axis=0)        # roll yield relative to the cross-section
    gross = x.abs().sum(axis=1).replace(0.0, np.nan)
    w = x.div(gross, axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def book_returns(returns: pd.DataFrame, roll_yield: pd.DataFrame, cost_bps: float = 5.0) -> pd.Series:
    """Net weekly return of the dollar-neutral carry book: weights applied to next-week returns, minus
    turnover cost (``cost_bps`` per unit traded). Roll yield is slow, so turnover — and the cost drag — is
    modest; commodity carry's tradability question is its crash tail, not its turnover."""
    w = carry_signal(roll_yield)
    gross = (w.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("carry")


def turnover(roll_yield: pd.DataFrame) -> float:
    """Average weekly one-way turnover (Σ|Δw|) — low for a slow roll-yield signal."""
    w = carry_signal(roll_yield)
    return float(w.diff().abs().sum(axis=1).mean())


def carry_premium_by_bucket(returns: pd.DataFrame, roll_yield: pd.DataFrame, n_buckets: int = 3) -> dict:
    """Annualised mean return of the top vs bottom roll-yield bucket, and the high-minus-low spread.

    Each week, sort commodities by lagged roll yield into ``n_buckets`` equal groups; hold the top
    (most-backwardated) and bottom (most-contangoed) as equal-weight baskets; report the annualised means
    and the long-minus-short spread. This is the cleanest read of "do backwardated commodities out-earn?"
    """
    ry = roll_yield.shift(1)
    top, bot = [], []
    for t in returns.index[1:]:
        row = ry.loc[t].dropna()
        if len(row) < n_buckets * 2:
            continue
        ranks = row.rank(method="first")
        q = len(row) / n_buckets
        hi = ranks[ranks > (n_buckets - 1) * q].index
        lo = ranks[ranks <= q].index
        top.append(returns.loc[t, hi].mean())
        bot.append(returns.loc[t, lo].mean())
    top = pd.Series(top); bot = pd.Series(bot)
    hi_ann = float(top.mean() * WEEKS_PER_YEAR)
    lo_ann = float(bot.mean() * WEEKS_PER_YEAR)
    return {"high_ann_pct": hi_ann * 100, "low_ann_pct": lo_ann * 100,
            "hml_ann_pct": (hi_ann - lo_ann) * 100}


def summary(returns: pd.Series, periods_per_year: int = WEEKS_PER_YEAR) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a periodic return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_periods")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_periods": int(len(r))}
