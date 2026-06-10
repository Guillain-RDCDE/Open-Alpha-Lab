"""Beat-7 worked complement — "does the inflation hedge actually pay when it's supposed to?"

The whole steelman for the inflation-hedge tilt (Neville et al. 2021) is *conditional*: real assets are
supposed to earn their keep **specifically in rising-inflation regimes**, and to be dead weight (or worse)
when inflation is falling. A book that pays the same in both regimes isn't hedging inflation — it's just
a long-real-assets bet wearing a macro costume.

So this complement **splits the sample by inflation regime** — months where inflation momentum is rising
vs falling (measured from the macro state, lagged so the split is causal) — and reports each book's
Sharpe / mean in each regime. The prediction we pre-register: the **inflation-hedge** book earns
materially more in the rising-inflation regime than in the falling one; the broader **macro-momentum**
book, which also rides growth, should be steadier across both.

House-standard *worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import _macro_momentum, book_returns, summary

MONTHS_PER_YEAR = 12


def inflation_regime(macro: pd.DataFrame, smooth: int = 3) -> pd.Series:
    """A label per month: ``+1`` rising-inflation, ``-1`` falling, from that month's inflation momentum.

    This is a *regime attribution*, not a trading signal: it sorts the book's already-earned returns by
    the realized inflation environment of the month they were earned (was inflation actually rising or
    falling?). That is the honest question the inflation hedge has to answer — "did you pay when inflation
    rose?" — so the contemporaneous label is the right one here (it uses no future information, only the
    same month's realized macro state, exactly what an ex-post regime study conditions on)."""
    mm = _macro_momentum(macro, smooth=smooth)["inflation"]
    return np.sign(mm).fillna(0.0).rename("infl_regime")


def regime_split(returns: pd.DataFrame, macro: pd.DataFrame, kind: str = "inflation",
                 cost_bps: float = 5.0, smooth: int = 3) -> pd.DataFrame:
    """Split a book's net return by inflation regime: rising vs falling inflation.

    Returns a frame indexed by regime (``rising`` / ``falling``) with the book's annualised Sharpe, mean
    return, and the month count in each regime. The inflation-hedge book should earn its premium in the
    *rising* row.
    """
    r = book_returns(returns, macro, kind=kind, cost_bps=cost_bps, smooth=smooth)
    reg = inflation_regime(macro, smooth=smooth).reindex(r.index).fillna(0.0)
    rows = {}
    for label, mask in (("rising", reg > 0), ("falling", reg < 0)):
        seg = r[mask]
        s = summary(seg)
        rows[label] = {"sharpe": s["sharpe"], "ann_return_pct": 100.0 * s["ann_return"],
                       "n_months": int(mask.sum())}
    out = pd.DataFrame(rows).T
    out.index.name = "inflation_regime"
    return out


def regime_split_both(returns: pd.DataFrame, macro: pd.DataFrame, cost_bps: float = 5.0,
                      smooth: int = 3) -> dict:
    """Convenience: the regime split for both books, so the notebook can show the contrast."""
    return {kind: regime_split(returns, macro, kind=kind, cost_bps=cost_bps, smooth=smooth)
            for kind in ("inflation", "macro_momentum")}
