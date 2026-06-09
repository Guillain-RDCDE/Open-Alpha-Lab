"""Worked extension (beat 7) — the tether drifts: the hedge ratio that defined the pair won't sit still.

A pairs trade assumes the relationship ``β`` between the two legs is stable. The honest follow-up is to
watch ``β`` over time: if the trailing hedge ratio wanders far, the "pair" you fit yesterday is a
different pair today, and the spread you're mean-reverting against is a moving target. That drift is the
slow way a pair dies — not a dramatic break, but a tether that quietly stretches until the trade is
betting on a relationship that no longer holds.

:func:`hedge_ratio_drift` measures how much the trailing ``β`` moves relative to its level;
:func:`oos_scan` runs the in-sample/out-of-sample split across a set of candidate pairs and reports how
many actually survive. Baked checks: a cointegrated synthetic pair has a *stable* ``β`` and survives
both halves; a spurious one has a wandering ``β`` and a collapsing second half.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .strategy import rolling_hedge_ratio
from .decompose import in_sample_vs_oos

TRADING_DAYS_PER_YEAR = 252


def hedge_ratio_drift(a: pd.Series, b: pd.Series, window: int = 252) -> dict:
    """How much does the trailing hedge ratio wander? Std of β relative to its mean.

    A stable (truly cointegrated) pair keeps a roughly constant β; a drifting β means the relationship
    is unstable and the spread is being measured against a moving anchor.
    """
    beta = rolling_hedge_ratio(a, b, window).dropna()
    if beta.empty:
        return {"beta_mean": np.nan, "beta_std": np.nan, "beta_rel_drift": np.nan,
                "beta_min": np.nan, "beta_max": np.nan}
    return {
        "beta_mean": float(beta.mean()),
        "beta_std": float(beta.std(ddof=1)),
        "beta_rel_drift": float(beta.std(ddof=1) / abs(beta.mean())) if abs(beta.mean()) > 1e-9 else np.inf,
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
    }


def oos_scan(closes: dict, cost_bps: float = 2.0, max_pairs: int = 30, **kw) -> pd.DataFrame:
    """In-sample vs out-of-sample Sharpe across candidate pairs from a basket of closes.

    For each pair (up to ``max_pairs``), reports the first-half and second-half Sharpe and whether it
    survives. The point: even the pairs that look good in the first half mostly fail the second.
    """
    names = list(closes.keys())
    rows = {}
    for a_tk, b_tk in itertools.islice(itertools.combinations(names, 2), max_pairs):
        res = in_sample_vs_oos(closes[a_tk], closes[b_tk], cost_bps=cost_bps, **kw)
        rows[f"{a_tk}/{b_tk}"] = {
            "first_half": res["first_half_sharpe"],
            "second_half": res["second_half_sharpe"],
            "survives": res["survives_oos"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "pair"
    return out
