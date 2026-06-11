"""Betting against beta — the factor, and the leverage it quietly runs on.

Frazzini & Pedersen (2014): rank assets by beta, go long the low-beta half and short the high-beta
half, **levering each leg to beta 1** so the spread is market-neutral. The low-beta leg therefore runs
at ``1/β_low`` ≈ 2-3× — and *that* is the whole debate: the strategy's headline Sharpe assumes you
borrow that leverage for free. Charge realistic financing and the edge is supposed to survive; here we
test whether it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """Trailing beta of ``asset`` to ``market`` = cov/var over ``window`` daily observations."""
    a, m = pd.Series(asset).astype(float), pd.Series(market).astype(float)
    return a.rolling(window).cov(m) / m.rolling(window).var()


def _monthly(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).resample("ME").prod() - 1.0


def bab_returns(
    assets: pd.DataFrame,
    market: pd.Series,
    financing_ann: float = 0.0,
    borrow_ann: float = 0.0,
    window: int = TRADING_DAYS,
    return_leverage: bool = False,
):
    """Monthly net return of the beta-neutral BAB book on a cross-section of assets.

    Each month: estimate trailing betas (as of the prior month-end), split at the median into a
    low-beta and a high-beta half, lever each equal-weight leg to beta 1 (``w = 1/β_leg``), and take
    long-low minus short-high. The long leg borrows ``w_low − 1`` notional at ``financing_ann``; the
    short leg pays ``borrow_ann`` on its notional. Returns the monthly net Series (and, optionally, the
    average long-leg leverage — the number the "free lunch" depends on).
    """
    a, m = pd.DataFrame(assets).astype(float), pd.Series(market).astype(float)
    betas = pd.DataFrame({c: rolling_beta(a[c], m, window) for c in a.columns})
    mret = _monthly(a)
    mdates = mret.index
    mbeta = betas.reindex(mdates, method="ffill")

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
        gross = wL * mret.loc[dt, low].mean() - wH * mret.loc[dt, high].mean()
        fin = max(wL - 1.0, 0.0) * (financing_ann / 12.0) + abs(wH) * (borrow_ann / 12.0)
        out[dt] = gross - fin
        levs.append(wL)
    s = pd.Series(out, name="bab")
    return (s, float(np.mean(levs)) if levs else np.nan) if return_leverage else s


def market_monthly(market: pd.Series) -> pd.Series:
    """The market's monthly return — the benchmark a 'market-neutral' book still has to justify."""
    return (_monthly(pd.Series(market).astype(float).to_frame("m"))["m"]).rename("market")


def summary(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }
