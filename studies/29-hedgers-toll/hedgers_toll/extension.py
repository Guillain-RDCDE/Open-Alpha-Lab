"""Worked extension (beat 7) — does the premium survive the parameter choices, and where does it live?

Two honest follow-ups. **(1) Robustness:** a hedging-pressure premium that only shows at one lookback
window is a fitted artefact, so we sweep the window over which the signal is normalised and read the
long-short Sharpe at each. **(2) Which leg:** is the premium in the *long* leg (commodities whose hedgers
are most net short — the speculator getting paid) or the *short* leg? :func:`window_sweep` traces the
factor Sharpe across normalisation windows; :func:`leg_split` compares the long-only-top book to the
long-short.

Baked check: on a panel with a real baked premium, the factor is positive across windows and the long
leg carries it; on the null, nothing.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import hp_returns, equal_weight, summary

WEEKS_PER_YEAR = 52


def window_sweep(returns: pd.DataFrame, hp: pd.DataFrame, windows=(52, 104, 156, 208, 260),
                 cost_bps: float = 10.0, periods_per_year: int = WEEKS_PER_YEAR, **kw) -> pd.DataFrame:
    """Long-short hedging-pressure Sharpe across the signal-normalisation window — robustness or fit."""
    rows = {}
    for w in windows:
        s = summary(hp_returns(returns, hp, window=w, cost_bps=cost_bps, long_short=True, **kw), periods_per_year)
        rows[w] = {"ls_sharpe": s["sharpe"], "ls_ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T; out.index.name = "window_weeks"
    return out


def leg_split(returns: pd.DataFrame, hp: pd.DataFrame, cost_bps: float = 10.0,
              periods_per_year: int = WEEKS_PER_YEAR, **kw) -> dict:
    """Long-only-top vs long-short vs the equal-weight basket — where the hedging premium lives."""
    lo = hp_returns(returns, hp, cost_bps=cost_bps, long_short=False, **kw)
    ls = hp_returns(returns, hp, cost_bps=cost_bps, long_short=True, **kw)
    ew = equal_weight(returns).reindex(lo.index)
    return {"long_only_top_sharpe": summary(lo, periods_per_year)["sharpe"],
            "long_short_sharpe": summary(ls, periods_per_year)["sharpe"],
            "basket_sharpe": summary(ew, periods_per_year)["sharpe"]}
