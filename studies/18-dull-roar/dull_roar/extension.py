"""Worked extension (beat 7) — the no-shorting test, and what survives it.

The headline study's one honest caveat — the reason the third axis is `BETA-TILT`, not "free alpha" —
is **the short leg**. The clean, beta-neutral anomaly leans on shorting the high-vol decile, and those
names (small, lottery-like, heavily-shorted) are exactly the ones that are expensive or impossible to
borrow. A real-money investor who can't short is left with the *long-only* low-vol book — which is
mostly a low-beta defensive tilt, not alpha.

So we make that caveat into a backtest. The anomaly's edge splits into two economically distinct
pieces, and this module isolates them:

    * **Defensive (long-only)** — overweight the calm decile. Needs **no shorting**, and it is where a
      real min-vol product (USMV, SPLV) lives. Its value is honest-but-bounded: lever it to beta 1 and
      compare to the index (:func:`dull_roar.decompose.beta_tilt_test`).
    * **Short-the-wild** — the high-vol leg's negative alpha. **Needs a borrow**, and it is the slice
      the literature says carries most of the anomaly (Ang et al. 2006) — and the slice a borrow fee
      eats first.

:func:`shorting_decomposition` measures the full beta-neutral BAB alpha, the long-only defensive
alpha, and the residual short-leg alpha, so ``alpha_full ≈ alpha_defensive + alpha_short`` splits the
edge. :func:`borrow_breakeven` finds the annual stock-loan fee at which the long-short's net Sharpe
crosses zero — how much shorting friction the dollar-neutral book can absorb before it stops paying.
Two baked-in checks pin the machine before the real run decides it:

    * **Null panel** (flat SML) → no alpha in either slice, at any borrow.
    * **Anomaly panel** → a real BAB alpha whose *harvestable* (long-only) share is the smaller,
      defensive part; the rest needs the short the real tape may not grant.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import compare
from .decompose import beta_neutral_bab, beta_tilt_test, leg_attribution

TRADING_DAYS_PER_YEAR = 252


def shorting_decomposition(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    borrow_bps_ann: float = 0.0,
    **build_kw,
) -> dict:
    """Split the beta-neutral anomaly alpha into a **no-shorting** defensive part and a **short** part.

    ``alpha_full`` is the beta-neutral BAB alpha (the whole effect). ``alpha_defensive`` is the
    long-only low-vol book's alpha after it is levered to beta 1 (the piece needing no borrow), and
    ``alpha_short`` is the remainder — the high-vol leg's negative alpha you can only capture by
    shorting. ``share_defensive`` is the fraction of the edge a long-only, no-shorting investor keeps.
    """
    bab = beta_neutral_bab(panel, market=market, cost_bps=cost_bps,
                           borrow_bps_ann=borrow_bps_ann, **build_kw)
    tilt = beta_tilt_test(panel, market=market, cost_bps=cost_bps, **build_kw)
    legs = leg_attribution(panel, market=market, cost_bps=cost_bps, **build_kw)

    alpha_full = bab["alpha_ann_pct"]
    # the long-only defensive alpha (low-vol book's own CAPM alpha) is the no-shorting slice
    alpha_defensive = legs["low_leg_alpha_pct"]
    alpha_short = alpha_full - alpha_defensive
    share_def = (alpha_defensive / alpha_full) if abs(alpha_full) > 1e-9 else float("nan")
    return {
        "alpha_full": float(alpha_full),
        "alpha_full_t": float(bab["alpha_t"]),
        "alpha_defensive": float(alpha_defensive),
        "alpha_defensive_t": float(legs["low_leg_alpha_t"]),
        "alpha_short": float(alpha_short),
        "share_defensive": float(share_def),
        "defensive_sharpe_gain": float(tilt["sharpe_gain"]),
        "defensive_excess_cagr_beta1_pct": float(tilt["excess_cagr_at_beta1_pct"]),
        "low_only_maxdd": float(tilt["low_only_maxdd"]),
        "market_maxdd": float(tilt["market_maxdd"]),
    }


def borrow_breakeven(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    hi_bps: float = 5000.0,
    tol_bps: float = 1.0,
    **build_kw,
) -> dict:
    """The annual short-leg borrow fee (bps) at which the naive long-short's net Sharpe hits zero.

    A bisection on ``borrow_bps_ann``: below the break-even the dollar-neutral book pays, above it the
    borrow eats the edge. A *low* break-even means the anomaly's tradable form is fragile to exactly
    the friction the high-vol leg attracts. Returns the break-even (or the bracket end if the Sharpe
    never crosses zero within ``[0, hi_bps]``).
    """
    def ls_sharpe(fee):
        return compare(panel, market=market, cost_bps=cost_bps, borrow_bps_ann=fee,
                       **build_kw)["long_short"]["sharpe"]

    s0 = ls_sharpe(0.0)
    if s0 <= 0:
        return {"breakeven_bps": 0.0, "sharpe_at_zero_borrow": float(s0), "crossed": True}
    if ls_sharpe(hi_bps) > 0:
        return {"breakeven_bps": float(hi_bps), "sharpe_at_zero_borrow": float(s0), "crossed": False}

    lo, hi = 0.0, hi_bps
    while hi - lo > tol_bps:
        mid = 0.5 * (lo + hi)
        if ls_sharpe(mid) > 0:
            lo = mid
        else:
            hi = mid
    return {"breakeven_bps": float(0.5 * (lo + hi)), "sharpe_at_zero_borrow": float(s0), "crossed": True}
