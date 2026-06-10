"""Beat-7 worked complement — "breadth is the lever, and the blend scheme barely matters."

The Fundamental Law (Grinold-Kahn): the information ratio of a combo scales with the *square root of the
number of independent bets*, IR ≈ IC · √breadth. So the natural follow-up to "the combo beats every
component" is: **how does the combo's Sharpe scale as you add components?** And a practical second
question: does the clever inverse-vol (risk-parity) blend actually beat the naive equal weight?

  * :func:`breadth_sweep` — include the first ``k`` components (in a fixed order) and report the combo
    Sharpe for ``k = 1, 2, …, N``, alongside √k for reference. A rising, concave curve is the Law.
  * :func:`scheme_comparison` — equal-weight vs risk-parity combo Sharpe, gross and net.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import book_returns, combine, summary


def breadth_sweep(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, scheme: str = "equal",
                  cost_bps: float = 0.0) -> pd.DataFrame:
    """Combo Sharpe as the number of included components grows from 1 to N (components added in dict
    order). Reports the Sharpe, its ratio to the 1-signal Sharpe, and √k for the Fundamental-Law eye."""
    names = list(signals)
    rows = {}
    base = None
    for k in range(1, len(names) + 1):
        sub = {nm: signals[nm] for nm in names[:k]}
        w = combine(sub, panel, scheme=scheme)
        sh = summary(book_returns(w, panel, cost_bps=cost_bps))["sharpe"]
        if base is None:
            base = sh
        rows[k] = {"components": ", ".join(names[:k]), "sharpe": sh,
                   "sharpe_ratio_to_1": sh / base if base else np.nan, "sqrt_k": np.sqrt(k)}
    out = pd.DataFrame(rows).T
    out.index.name = "n_signals"
    return out


def scheme_comparison(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, cost_bps: float = 0.0
                      ) -> pd.DataFrame:
    """Equal-weight vs risk-parity combo: Sharpe, CAGR, vol — gross or at ``cost_bps``."""
    rows = {}
    for scheme in ("equal", "risk_parity"):
        w = combine(signals, panel, scheme=scheme)
        s = summary(book_returns(w, panel, cost_bps=cost_bps))
        rows[scheme] = {"sharpe": s["sharpe"], "cagr": s["cagr"], "vol_ann": s["vol_ann"]}
    out = pd.DataFrame(rows).T
    out.index.name = "scheme"
    return out
