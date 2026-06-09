"""Worked extension (beat 7) — stack the two crash defences: residualise *and* vol-manage.

Residual momentum sheds the systematic (beta-driven) part of the crash. [Study 24](../../24-stampede/)'s
beat 7 showed vol-scaling sheds the rest (the crash clusters in high-vol regimes). The natural question:
do they *stack*? This complement builds three books — total-WML, residual-WML, and vol-managed
residual-WML — and lines up their crash profiles, so you can read the progressive taming of the tail.

:func:`vol_managed` scales any WML stream by inverse trailing vol; :func:`defence_stack` compares the
three. Baked check: each defence shrinks the drawdown without killing the premium.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import wml_returns, summary

TRADING_DAYS_PER_YEAR = 252


def vol_managed(stream: pd.Series, target_vol_ann: float = 0.10, vol_window: int = 63,
                max_leverage: float = 3.0) -> pd.Series:
    """Scale a return stream to a constant risk target by its own *trailing* (past-only) volatility."""
    target_daily = target_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    lev = (target_daily / stream.rolling(vol_window).std(ddof=1).shift(1)).clip(upper=max_leverage)
    return (lev * stream).dropna().rename("managed")


def _profile(r, periods_per_year):
    s = summary(r, periods_per_year)
    monthly = ((1.0 + r).resample("ME").prod() - 1.0).dropna()
    return {"sharpe": s["sharpe"], "skew": float(monthly.skew()),
            "worst_month_pct": float(monthly.min() * 100.0), "max_drawdown_pct": float(s["max_drawdown"] * 100.0)}


def defence_stack(panel: pd.DataFrame, market: pd.Series | None = None, cost_bps: float = 5.0,
                  periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Total-WML vs residual-WML vs vol-managed residual-WML — progressive crash taming.

    Reads the three books' Sharpe, skew, worst month and drawdown. The story: residualising removes the
    beta-driven crash, vol-scaling removes the regime-clustered remainder, and the premium survives.
    """
    tot = wml_returns(panel, market, residual=False, cost_bps=cost_bps, **kw)
    res = wml_returns(panel, market, residual=True, cost_bps=cost_bps, **kw)
    res_vm = vol_managed(res)
    idx = res.index.intersection(tot.index)
    return {
        "total": _profile(tot.reindex(idx), periods_per_year),
        "residual": _profile(res.reindex(idx), periods_per_year),
        "residual_vol_managed": _profile(res_vm.reindex(res_vm.index.intersection(idx)), periods_per_year),
    }
