"""The teardown that earns the stamps — the carry premium is real, and it crashes.

Three legs:

  1. :func:`premium_tstat` — is the carry portfolio's mean return reliably positive (Newey-West)? A
     large *t* is the `REAL` signal — UIRP fails and the high-rate currencies do out-earn.
  2. :func:`crash_profile` — the catch, and the study's namesake. Carry has a fat **negative** skew: it
     earns a steady drip and then loses a fortune in a global risk-off (1998, 2008). We measure the
     skew, the worst month, and the drawdown — "picking up nickels in front of a steamroller".
  3. :func:`downside_concentration` — how much of the total loss comes from the worst few months, the
     fingerprint of a crash-prone, not a diffuse, risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import carry_returns, summary

MONTHS_PER_YEAR = 12


def premium_tstat(xret: pd.DataFrame, rates: pd.DataFrame, cost_bps: float = 10.0,
                  periods_per_year: int = MONTHS_PER_YEAR, **kw) -> dict:
    """Newey-West t-stat that the carry portfolio's monthly mean return is non-zero."""
    r = carry_returns(xret, rates, cost_bps=cost_bps, **kw).to_numpy()
    n = r.size; mu = r.mean(); e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_ann_pct": float(mu * periods_per_year * 100.0),
            "t_stat": float(mu / se) if se > 0 else np.nan, "n_months": int(n)}


def crash_profile(xret: pd.DataFrame, rates: pd.DataFrame, cost_bps: float = 10.0,
                  periods_per_year: int = MONTHS_PER_YEAR, **kw) -> dict:
    """The steamroller: the carry return's negative skew, worst month and worst drawdown."""
    r = carry_returns(xret, rates, cost_bps=cost_bps, **kw)
    s = summary(r, periods_per_year)
    worst5 = r.nsmallest(5)
    return {
        "sharpe": s["sharpe"], "skew": float(r.skew()),
        "worst_month_pct": float(r.min() * 100.0),
        "worst5_months_mean_pct": float(worst5.mean() * 100.0),
        "max_drawdown_pct": float(s["max_drawdown"] * 100.0),
        "best_month_pct": float(r.max() * 100.0),
        "n_months": int(len(r)),
    }


def downside_concentration(xret: pd.DataFrame, rates: pd.DataFrame, cost_bps: float = 10.0, k: int = 5, **kw) -> dict:
    """Share of the *total negative* return contributed by the worst ``k`` months — crash-prone if high."""
    r = carry_returns(xret, rates, cost_bps=cost_bps, **kw)
    neg = r[r < 0]
    worst_k = r.nsmallest(k).sum()
    total_neg = neg.sum()
    return {"worst_k_share_of_losses": float(worst_k / total_neg) if total_neg < 0 else np.nan,
            "k": int(k), "n_negative_months": int(len(neg))}
