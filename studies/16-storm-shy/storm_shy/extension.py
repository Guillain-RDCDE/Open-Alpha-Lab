"""Worked extension (beat 7) — the no-borrowing test, and what survives it.

The headline study's one honest caveat — the reason the third axis is `RISK-MANAGED`, not "free
lunch" — is **leverage**. To hold a constant risk target the overlay must *gear up* in calm regimes
(average leverage > 1), and a financing-constrained book (no margin, a long-only mandate, a leverage
cap) can't always do that. The Cederburg–O'Doherty–Wang–Yan (2020) critique is exactly this: a
Sharpe gain that *needs* leverage is not the same as an investor gain.

So we make that caveat into a backtest. The overlay's edge splits into two economically distinct
pieces, and this module isolates them by **capping leverage**:

    * **De-risk** — cutting exposure in storms. Needs **no borrowing** (weights only ever fall below
      1), and it is where the drawdown reduction lives. This is the piece a constrained book keeps.
    * **Lever-up-calm** — adding exposure in quiet regimes to keep risk on target. **Needs leverage**,
      and it is the contested slice Cederburg targets.

:func:`gain_decomposition` measures the Sharpe gain at a leverage cap of **1.0** (de-risk only) and
at a high cap (~uncapped), so ``gain_full = gain_derisk + gain_leverage`` splits the edge cleanly.
:func:`leverage_cap_sweep` traces the whole curve. Two baked-in checks pin the machine before the
real run decides it:

    * **Flat-vol tape** → no regime, so no gain at *any* cap (the null).
    * **Clustered tape** → a meaningful share of the gain survives at cap 1.0 (de-risking works
      without borrowing); the rest is the leverage slice. Whether that survives on the real tape is
      left to :mod:`examples.extension` on SPY/QQQ.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import compare
from .decompose import certainty_equivalent, equal_risk_return

TRADING_DAYS_PER_YEAR = 252


def leverage_cap_sweep(
    returns: pd.Series,
    caps=(1.0, 1.25, 1.5, 2.0, 3.0, 10.0),
    target_vol_ann: float = 0.12,
    window: int = 21,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Net Sharpe gain (and leverage/drawdown) as the leverage cap is relaxed from 1.0 upward.

    Everything else is held fixed, so the *only* thing changing down the table is how much the book
    is allowed to borrow. ``cap = 1.0`` is the no-borrowing, de-risk-only investor; a high cap is the
    unconstrained Moreira–Muir overlay. The gap between the two rows is the leverage-timing slice.
    """
    rows = {}
    for cap in caps:
        c = compare(returns, target_vol_ann=target_vol_ann, window=window,
                    max_leverage=cap, cost_bps=cost_bps)
        rows[cap] = {
            "sharpe_gain": c["sharpe_gain_net"],
            "managed_sharpe": c["managed_net"]["sharpe"],
            "avg_leverage": c["avg_leverage"],
            "frac_capped": c["frac_capped"],
            "managed_maxdd": c["managed_net"]["max_drawdown"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "max_leverage"
    return out


def gain_decomposition(
    returns: pd.Series,
    derisk_cap: float = 1.0,
    full_cap: float = 10.0,
    target_vol_ann: float = 0.12,
    window: int = 21,
    cost_bps: float = 1.0,
    gamma: float = 5.0,
) -> dict:
    """Split the overlay's Sharpe gain into a **de-risk** part (no borrowing) and a **leverage** part.

    ``gain_derisk`` is the net Sharpe gain at leverage cap ``derisk_cap`` (1.0 — the book only ever
    cuts exposure); ``gain_full`` is the gain at ``full_cap`` (~uncapped). Their difference is
    ``gain_leverage`` — the contested slice that *requires* gearing up in calm regimes. Also reports
    the share of the gain that survives with no borrowing, the drawdown reduction at cap 1.0 (which
    needs no leverage), and the no-borrow CRRA certainty-equivalent — the number a constrained
    investor actually banks.
    """
    c_d = compare(returns, target_vol_ann=target_vol_ann, window=window,
                  max_leverage=derisk_cap, cost_bps=cost_bps)
    c_f = compare(returns, target_vol_ann=target_vol_ann, window=window,
                  max_leverage=full_cap, cost_bps=cost_bps)
    gd, gf = c_d["sharpe_gain_net"], c_f["sharpe_gain_net"]

    kw = dict(target_vol_ann=target_vol_ann, window=window, max_leverage=derisk_cap, cost_bps=cost_bps)
    ce_d = certainty_equivalent(returns, gamma=gamma, **kw)
    er_d = equal_risk_return(returns, **kw)

    return {
        "gain_derisk": float(gd),
        "gain_full": float(gf),
        "gain_leverage": float(gf - gd),
        "share_derisk": float(gd / gf) if abs(gf) > 1e-12 else float("nan"),
        "derisk_avg_leverage": float(c_d["avg_leverage"]),
        "full_avg_leverage": float(c_f["avg_leverage"]),
        "buyhold_maxdd": float(c_d["buy_hold"]["max_drawdown"]),
        "derisk_maxdd": float(c_d["managed_net"]["max_drawdown"]),
        "ce_gain_derisk_pct": float(ce_d["ce_gain_pct"]),
        "excess_cagr_derisk_pct": float(er_d["excess_cagr_pct"]),
    }
