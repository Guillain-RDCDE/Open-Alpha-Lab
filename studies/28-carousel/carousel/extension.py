"""Worked extension (beat 7) — was holding 3 sectors the lucky pick?

A rotation rule that only beats the equal-weight basket at one hand-picked number of sectors is a
data-mined artefact. So we sweep ``top_k`` from 1 (most concentrated) to 6 (half the universe) and read
the rotation-minus-equal-weight Sharpe at each. If sector momentum were a robust edge, most ``k`` would
help; if it's noise, the win-rate hovers around a coin flip and the best ``k`` is just the luckiest.

:func:`topk_sweep` returns the rotation Sharpe and the gain over equal-weight for each ``top_k``.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import compare

TRADING_DAYS_PER_YEAR = 252


def topk_sweep(panel: pd.DataFrame, ks=(1, 2, 3, 4, 5, 6), cost_bps: float = 3.0,
               periods_per_year: int = TRADING_DAYS_PER_YEAR, **kw) -> dict:
    """Rotation Sharpe and gain over the equal-weight basket for each ``top_k`` — robustness or cherry-pick."""
    rows = {}
    gains = []
    for k in ks:
        cmp = compare(panel, top_k=k, cost_bps=cost_bps, periods_per_year=periods_per_year, **kw)
        rows[k] = {"rotation_sharpe": cmp["rotation"]["sharpe"], "ew_sharpe": cmp["equal_weight"]["sharpe"],
                   "gain_over_ew": cmp["rotation_minus_ew_sharpe"]}
        gains.append(cmp["rotation_minus_ew_sharpe"])
    matrix = pd.DataFrame(rows).T
    matrix.index.name = "top_k"
    gains = np.array(gains)
    return {"matrix": matrix, "frac_beat_ew": float((gains > 0).mean()),
            "mean_gain": float(gains.mean()), "best_k": int(ks[int(np.argmax(gains))]), "best_gain": float(gains.max())}
