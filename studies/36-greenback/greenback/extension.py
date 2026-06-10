"""Beat-7 worked complement — the carry⊕momentum diversification, and the carry crash.

The base run builds the three sleeves; this complement makes the *combo* argument explicit and shows the
crash it is meant to dull. Two pieces:

  * :func:`diversification` — the headline of the study: the combo's Sharpe beats *either* standalone leg,
    and the two legs' returns are **low- or negatively-correlated** (carry and momentum pay at different
    times — Asness–Moskowitz–Pedersen 2013; Koijen et al "Carry" 2018). Reports each leg's Sharpe, the
    combo Sharpe, and the carry↔momentum correlation.
  * :func:`carry_crash` — the steamroller signature on the synthetic: the carry book's negative skew and
    its worst months, and how the combo's worst months compare (does momentum cushion them?).

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import carry_returns, combine, momentum_returns, summary

MONTHS_PER_YEAR = 12


def diversification(xret: pd.DataFrame, rate_diffs: pd.DataFrame, cost_bps: float = 10.0,
                    w_carry: float = 0.5) -> dict:
    """Carry vs momentum vs the combo — Sharpes, the leg correlation, and the combo's Sharpe uplift."""
    carry = carry_returns(xret, rate_diffs, cost_bps=cost_bps)
    mom = momentum_returns(xret, cost_bps=cost_bps)
    combo = combine(carry, mom, w_carry=w_carry)
    idx = carry.index.intersection(mom.index)
    corr = float(carry.reindex(idx).corr(mom.reindex(idx)))
    sc, sm, scombo = (summary(x)["sharpe"] for x in (carry, mom, combo))
    return {"carry_sharpe": sc, "momentum_sharpe": sm, "combo_sharpe": scombo,
            "leg_correlation": corr, "combo_beats_legs": bool(scombo > max(sc, sm)),
            "uplift_vs_best_leg": float(scombo - max(sc, sm)), "n_months": int(len(idx))}


def carry_crash(xret: pd.DataFrame, rate_diffs: pd.DataFrame, cost_bps: float = 10.0,
                w_carry: float = 0.5, n_worst: int = 5) -> dict:
    """The steamroller signature: carry's negative skew / worst months, and whether the combo cushions them."""
    carry = carry_returns(xret, rate_diffs, cost_bps=cost_bps)
    mom = momentum_returns(xret, cost_bps=cost_bps)
    combo = combine(carry, mom, w_carry=w_carry)
    idx = carry.index.intersection(combo.index)
    c, k = carry.reindex(idx), combo.reindex(idx)
    worst_carry_months = c.nsmallest(n_worst)
    # what the combo earned in carry's worst months (does momentum offset the crash?)
    combo_in_carry_crashes = float(k.reindex(worst_carry_months.index).mean() * 100.0)
    return {"carry_skew": float(c.skew()), "combo_skew": float(k.skew()),
            "carry_worst_month_pct": float(c.min() * 100.0),
            "combo_worst_month_pct": float(k.min() * 100.0),
            "carry_worst5_mean_pct": float(worst_carry_months.mean() * 100.0),
            "combo_in_carry_worst5_pct": combo_in_carry_crashes,
            "carry_max_dd_pct": float(summary(c)["max_drawdown"] * 100.0),
            "combo_max_dd_pct": float(summary(k)["max_drawdown"] * 100.0)}
