"""Beat-7 worked complement — "does adding a momentum sleeve diversify the carry book?"

Carry and time-series momentum are the two classic commodity premia, and they are known to be lowly
correlated — carry leans on the term structure, momentum on the price trend, and they tend to pay off in
different regimes (Koijen-Moskowitz-Pedersen-Vrugt 2018; Moskowitz-Ooi-Pedersen 2012). So the natural
question after measuring carry alone: **does blending in a commodity time-series-momentum sleeve raise the
combined book's risk-adjusted return above carry standalone?**

  * :func:`momentum_signal` — a cross-sectional 26-week (≈6-month) return rank, dollar-neutral, lagged: a
    standard commodity trend sleeve.
  * :func:`combine` — a 50/50 risk blend of the carry book and the momentum book; reports each leg's
    Sharpe, the blend's Sharpe, and the leg-to-leg correlation. The diversification test: blend Sharpe >
    max(leg Sharpes) when the legs are lowly correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, carry_signal, summary


def momentum_signal(returns: pd.DataFrame, lookback: int = 26) -> pd.DataFrame:
    """Dollar-neutral cross-sectional momentum weights: the cross-sectionally demeaned trailing
    ``lookback``-week return, normalised to gross 1, lagged one week. Long the trenders, short the laggers."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    trail = prices / prices.shift(lookback) - 1.0
    x = trail.sub(trail.mean(axis=1), axis=0)
    gross = x.abs().sum(axis=1).replace(0.0, np.nan)
    w = x.div(gross, axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def momentum_returns(returns: pd.DataFrame, lookback: int = 26, cost_bps: float = 5.0) -> pd.Series:
    """Net weekly return of the dollar-neutral commodity momentum sleeve."""
    w = momentum_signal(returns, lookback=lookback)
    gross = (w.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("momentum")


def combine(returns: pd.DataFrame, roll_yield: pd.DataFrame, lookback: int = 26, cost_bps: float = 5.0,
            w_carry: float = 0.5) -> dict:
    """Risk view of carry, momentum, and a ``w_carry``/``1-w_carry`` blend.

    Returns each leg's Sharpe, the blended book's Sharpe, the leg correlation, and the three return
    series. The diversification claim: a low (or negative) carry-momentum correlation lets the blend's
    Sharpe exceed either standalone leg.
    """
    carry = book_returns(returns, roll_yield, cost_bps=cost_bps)
    mom = momentum_returns(returns, lookback=lookback, cost_bps=cost_bps)
    df = pd.concat([carry, mom], axis=1).dropna()
    blend = (w_carry * df["carry"] + (1.0 - w_carry) * df["momentum"]).rename("blend")
    sc = summary(df["carry"]); sm = summary(df["momentum"]); sb = summary(blend)
    corr = float(df["carry"].corr(df["momentum"]))
    return {"carry_sharpe": sc["sharpe"], "momentum_sharpe": sm["sharpe"], "blend_sharpe": sb["sharpe"],
            "correlation": corr,
            "carry": df["carry"], "momentum": df["momentum"], "blend": blend}
