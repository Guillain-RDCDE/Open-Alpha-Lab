"""Worked extension (beat 7) — the bias survives everything you'd throw at it (except going causal).

A practitioner who got the spectacular two-sided result would try the usual robustness checks — charge
more costs, vary the smoothing ``lam`` — and the strategy would *still* look great, because the
look-ahead bias dwarfs all of them. That's the trap's danger: every check passes except the one that
matters (use only past data). So we sweep ``lam`` and the cost, and show the two-sided Sharpe stays
large while the one-sided stays ~0 on a random walk — the bias is not a tuning artefact, it is the
filter peeking, at every setting.

:func:`lam_sweep` traces both books' Sharpe across smoothing levels; :func:`cost_robustness` shows the
two-sided edge laughs off realistic costs. Baked check: on the random-walk null the two-sided book is
large and the one-sided ~0 at *every* lam — only causality kills it.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import reversion_returns, summary

TRADING_DAYS_PER_YEAR = 252


def lam_sweep(close: pd.Series, lams=(1e4, 1e5, 1e6, 1e7), cost_bps: float = 1.0, window: int = 252) -> pd.DataFrame:
    """Two-sided vs one-sided Sharpe across HP smoothing ``lam`` — the bias persists at every setting."""
    rows = {}
    for lam in lams:
        two = summary(reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=False))
        one = summary(reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=True, window=window))
        rows[lam] = {"two_sided_sharpe": two["sharpe"], "one_sided_sharpe": one["sharpe"]}
    out = pd.DataFrame(rows).T
    out.index.name = "lam"
    return out


def cost_robustness(close: pd.Series, costs=(0, 1, 2, 5, 10), lam: float = 1e6, window: int = 252) -> pd.DataFrame:
    """Two-sided vs one-sided Sharpe as cost rises — the look-ahead edge shrugs off realistic costs."""
    rows = {}
    for c in costs:
        two = summary(reversion_returns(close, cost_bps=c, lam=lam, causal=False))
        one = summary(reversion_returns(close, cost_bps=c, lam=lam, causal=True, window=window))
        rows[c] = {"two_sided_sharpe": two["sharpe"], "one_sided_sharpe": one["sharpe"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
