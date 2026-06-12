"""Betting against beta — the factor, and the leverage it quietly runs on.

Frazzini & Pedersen (2014): rank assets by beta, go long the low-beta half and short the high-beta
half, **levering each leg to beta 1** so the spread is market-neutral. The low-beta leg therefore runs
at ``1/β_low`` ≈ 2-3× — and *that* is the whole debate: the strategy's headline Sharpe assumes the
leverage is financed at the risk-free rate with no frictions. We price it with an explicit
self-financing ledger and ask what survives.

**The ledger** (capital = 1, long ``w_L`` of the low-beta leg, short ``w_H`` of the high-beta leg):

  * the capital itself buys the first unit of the long leg; the slice above it, ``w_L − 1``, is
    borrowed at ``rf + financing_spread``;
  * the short proceeds ``w_H`` sit in the margin account earning the rebate ``rf − borrow_fee``;
  * every dollar is charged or credited exactly once (the Study 30 House-Edge discipline).

Total account return = ``rf + w_L·(r_L − rf) − w_H·(r_H − rf) − (w_L−1)·spread − w_H·fee``. The part
after ``rf`` is the book's **excess-of-cash** return — at zero spreads it is *exactly* the textbook
BAB factor, so "gross" here means "frictionless self-financed", not "borrow at 0%". Every Sharpe in
the study is excess-of-cash, BAB and market alike, so the race is like-for-like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Trailing beta of ``asset`` to ``market`` = cov/var over ``window`` daily observations."""
    a, m = pd.Series(asset).astype(float), pd.Series(market).astype(float)
    return a.rolling(window).cov(m) / m.rolling(window).var()


def _monthly(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).resample("ME").prod() - 1.0


def bab_returns(
    assets: pd.DataFrame,
    market: pd.Series,
    rf_monthly: pd.Series | None = None,
    financing_spread_ann: float = 0.0,
    borrow_fee_ann: float = 0.0,
    window: int = TRADING_DAYS,
    return_leverage: bool = False,
):
    """Monthly **excess-of-cash** return of the self-financed BAB book on a cross-section of assets.

    Each month: estimate trailing betas (as of the prior month-end), split at the median into a
    low-beta and a high-beta half, lever each equal-weight leg to beta 1 (``w = 1/β_leg``), and take
    long-low minus short-high **with each leg measured in excess of cash** — the Frazzini-Pedersen
    construction. The frictions sit on top of the risk-free rate, never instead of it: the borrowed
    slice ``w_L − 1`` pays ``financing_spread_ann`` over ``rf``, the short notional ``w_H`` gives up
    ``borrow_fee_ann`` of its rebate. ``rf_monthly`` is a monthly cash-return series (``None`` → 0,
    for the synthetic worlds). Returns the monthly excess Series (and, optionally, the average
    long-leg leverage — the number the "free lunch" depends on).
    """
    a, m = pd.DataFrame(assets).astype(float), pd.Series(market).astype(float)
    betas = pd.DataFrame({c: rolling_beta(a[c], m, window) for c in a.columns})
    mret = _monthly(a)
    mdates = mret.index
    mbeta = betas.reindex(mdates, method="ffill")
    rf = (pd.Series(0.0, index=mdates) if rf_monthly is None
          else pd.Series(rf_monthly).astype(float).reindex(mdates).fillna(0.0))

    out, levs = {}, []
    for i in range(1, len(mdates)):
        dt, prev = mdates[i], mdates[i - 1]
        b = mbeta.loc[prev].dropna()
        if len(b) < 4:
            continue
        med = b.median()
        low, high = b[b <= med].index, b[b > med].index
        bL, bH = b[low].mean(), b[high].mean()
        if bL <= 0 or bH <= 0:
            continue
        wL, wH = 1.0 / bL, 1.0 / bH                       # lever each leg to beta 1
        rf_t = rf.loc[dt]
        gross = wL * (mret.loc[dt, low].mean() - rf_t) - wH * (mret.loc[dt, high].mean() - rf_t)
        cost = max(wL - 1.0, 0.0) * (financing_spread_ann / MONTHS) + abs(wH) * (borrow_fee_ann / MONTHS)
        out[dt] = gross - cost
        levs.append(wL)
    s = pd.Series(out, name="bab")
    return (s, float(np.mean(levs)) if levs else np.nan) if return_leverage else s


def market_monthly(market: pd.Series) -> pd.Series:
    """The market's monthly return — the benchmark a 'market-neutral' book still has to justify."""
    return (_monthly(pd.Series(market).astype(float).to_frame("m"))["m"]).rename("market")


def summary(returns: pd.Series, periods_per_year: int = MONTHS,
            rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series.

    **Sharpe convention**: raw (``mean/std``) when ``rf`` is None; with ``rf`` (a monthly cash-return
    series) it is the standard **excess-of-cash** Sharpe. Pass the *same* ``rf`` to every line of a
    race so the comparison is like-for-like. CAGR / vol / max-drawdown always describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    ex = r if rf is None else (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
