"""Worked extension (beat 7) — risk-managed momentum: can vol-scaling tame the crash?

The headline catch is the crash: WML's fat negative tail (loser rebounds) wrecks the long-run record.
Barroso & Santa-Clara (2015) showed the crashes are *forecastable* — they cluster when the WML factor's
own recent volatility is high — so scaling the position by the inverse of trailing WML vol (a vol-target
overlay, exactly [Study 16](../../16-storm-shy/)'s machinery) cuts the worst drawdowns and lifts the
Sharpe. This complement makes that concrete: it builds the vol-managed WML and compares its crash
profile and Sharpe to the plain factor.

:func:`vol_managed_wml` returns the scaled stream (past-only); :func:`crash_comparison` puts the plain
and managed factors side by side on Sharpe, skew and drawdown. Baked check: on a momentum tape the
managed factor keeps the alpha while shrinking the tail; on the null both are ~0.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import books, summary

TRADING_DAYS_PER_YEAR = 252


def vol_managed_wml(panel: pd.DataFrame, target_vol_ann: float = 0.12, vol_window: int = 63,
                    max_leverage: float = 3.0, cost_bps: float = 5.0, **kw) -> pd.Series:
    """The WML factor scaled to a constant risk target by its own *trailing* (past-only) volatility."""
    wml = books(panel, cost_bps=cost_bps, **kw)["wml"]
    target_daily = target_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    fc = wml.rolling(vol_window).std(ddof=1).shift(1)
    lev = (target_daily / fc).clip(upper=max_leverage)
    return (lev * wml).dropna().rename("wml_managed")


def crash_comparison(panel: pd.DataFrame, cost_bps: float = 5.0,
                     periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Plain WML vs vol-managed WML — Sharpe, monthly skew and worst drawdown side by side."""
    plain = books(panel, cost_bps=cost_bps, **kw)["wml"]
    managed = vol_managed_wml(panel, cost_bps=cost_bps, **kw)
    idx = plain.index.intersection(managed.index)
    plain, managed = plain.reindex(idx), managed.reindex(idx)

    def _crash(r):
        s = summary(r, periods_per_year)
        monthly = ((1.0 + r).resample("ME").prod() - 1.0).dropna()
        return {"sharpe": s["sharpe"], "skew": float(monthly.skew()),
                "max_drawdown_pct": float(s["max_drawdown"] * 100.0),
                "worst_month_pct": float(monthly.min() * 100.0)}

    return {"plain": _crash(plain), "managed": _crash(managed),
            "sharpe_gain": float(_crash(managed)["sharpe"] - _crash(plain)["sharpe"])}
