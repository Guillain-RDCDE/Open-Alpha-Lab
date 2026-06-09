"""Worked extension (beat 7) — can risk management get you off the steamroller's path?

Carry's crashes cluster in high-volatility, risk-off regimes — so the same insight that earned
[Study 16](../../16-storm-shy/) the desk's only green applies: scale the position by the inverse of the
carry book's own recent volatility, and you cut exposure going into the storm. This complement builds the
vol-managed carry book and compares its crash profile to the plain one — does it dull the steamroller, or
just rearrange the bruises?

:func:`vol_managed_carry` scales the carry stream by inverse trailing vol; :func:`crash_comparison` puts
plain and managed side by side on Sharpe, skew, worst month and drawdown.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import carry_returns, summary

MONTHS_PER_YEAR = 12


def vol_managed_carry(xret: pd.DataFrame, rates: pd.DataFrame, target_vol_ann: float = 0.08,
                      vol_window: int = 12, max_leverage: float = 3.0, cost_bps: float = 10.0, **kw) -> pd.Series:
    """The carry stream scaled to a constant risk target by its own *trailing* (past-only) volatility."""
    r = carry_returns(xret, rates, cost_bps=cost_bps, **kw)
    target_m = target_vol_ann / np.sqrt(MONTHS_PER_YEAR)
    lev = (target_m / r.rolling(vol_window).std(ddof=1).shift(1)).clip(upper=max_leverage)
    return (lev * r).dropna().rename("carry_managed")


def _profile(r, periods_per_year):
    s = summary(r, periods_per_year)
    return {"sharpe": s["sharpe"], "skew": float(r.skew()),
            "worst_month_pct": float(r.min() * 100.0), "max_drawdown_pct": float(s["max_drawdown"] * 100.0)}


def crash_comparison(xret: pd.DataFrame, rates: pd.DataFrame, cost_bps: float = 10.0,
                     periods_per_year: int = MONTHS_PER_YEAR, **kw) -> dict:
    """Plain carry vs vol-managed carry — Sharpe, skew, worst month and drawdown side by side."""
    plain = carry_returns(xret, rates, cost_bps=cost_bps, **kw)
    managed = vol_managed_carry(xret, rates, cost_bps=cost_bps, **kw)
    idx = plain.index.intersection(managed.index)
    return {"plain": _profile(plain.reindex(idx), periods_per_year),
            "managed": _profile(managed.reindex(idx), periods_per_year)}
