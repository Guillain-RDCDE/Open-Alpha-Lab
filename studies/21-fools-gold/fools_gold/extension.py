"""Worked extension (beat 7) — the parameter-robustness test: was 50/200 just the lucky pick?

The headline runs one crossover (50/200). The honest follow-up: a folk rule that only works at one
hand-picked parameter pair is a data-mined artefact, not an edge. So we sweep a grid of (fast, slow)
pairs and ask how many beat buy-and-hold on a net-of-cost Sharpe basis — and by how much. If the
crossover carried real information, most sensible pairs would help; if it's noise, the win rate hovers
around a coin flip and the best pair is just the luckiest draw.

:func:`param_grid` returns the net Sharpe gain over buy-and-hold for every (fast, slow) pair and the
fraction that beat it. Two baked checks: on a trending tape a *majority* of pairs help (there is a real
trend to ride), on the driftless null essentially none do (no trend, only cost).

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import compare

TRADING_DAYS_PER_YEAR = 252

FASTS = (10, 20, 50, 100)
SLOWS = (50, 100, 150, 200, 250)


def param_grid(close: pd.Series, fasts=FASTS, slows=SLOWS, cost_bps: float = 2.0) -> dict:
    """Net Sharpe gain over buy-and-hold for every valid (fast < slow) crossover pair.

    Returns the gain matrix (fasts x slows), the fraction of pairs that beat buy-and-hold, the best
    pair and its gain, and the average gain — the spread of outcomes tells you whether 50/200 was
    informative or just a fortunate cell.
    """
    rows = {}
    gains = []
    best = (None, -np.inf)
    for f in fasts:
        rows[f] = {}
        for s in slows:
            if f >= s:
                rows[f][s] = np.nan
                continue
            g = compare(close, cost_bps=cost_bps, fast=f, slow=s)["sharpe_gain"]
            rows[f][s] = g
            gains.append(g)
            if g > best[1]:
                best = ((f, s), g)
    matrix = pd.DataFrame(rows).T
    matrix.index.name = "fast"
    gains = np.array(gains)
    return {
        "matrix": matrix,
        "frac_beat_buy_hold": float((gains > 0).mean()),
        "best_pair": best[0],
        "best_gain": float(best[1]),
        "mean_gain": float(gains.mean()),
        "n_pairs": int(gains.size),
    }
