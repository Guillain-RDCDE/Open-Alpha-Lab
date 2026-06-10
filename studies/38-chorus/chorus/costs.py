"""Costs and the standalone-vs-combo comparison — the two headline tables of the study.

The combo's whole appeal is a higher *gross* Sharpe than any single component. Beat 6 then asks the only
question that matters for trading: does the blend survive its own turnover? A combo of three signals that
each rebalance daily turns over at least as much as its busiest component, so cost erodes it fast. These
helpers produce (a) the standalone-vs-combo Sharpe table that proves "whole > parts", and (b) the cost
sweep that shows where the blend dies.
"""

from __future__ import annotations

import pandas as pd

from .strategy import avg_pairwise_corr, book_returns, combine, summary, turnover


def standalone_vs_combo(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, cost_bps: float = 0.0
                        ) -> pd.DataFrame:
    """One row per component plus two combo rows (equal-weight and risk-parity): Sharpe, CAGR, vol and
    turnover at ``cost_bps``. The centrepiece — the combo Sharpe should exceed every component's."""
    rows = {}
    for nm, w in signals.items():
        s = summary(book_returns(w, panel, cost_bps=cost_bps))
        rows[nm] = {"sharpe": s["sharpe"], "cagr": s["cagr"], "vol_ann": s["vol_ann"],
                    "turnover": turnover(w)}
    for scheme, label in [("equal", "COMBO (equal-wt)"), ("risk_parity", "COMBO (risk-parity)")]:
        w = combine(signals, panel, scheme=scheme)
        s = summary(book_returns(w, panel, cost_bps=cost_bps))
        rows[label] = {"sharpe": s["sharpe"], "cagr": s["cagr"], "vol_ann": s["vol_ann"],
                       "turnover": turnover(w)}
    out = pd.DataFrame(rows).T
    out.index.name = "book"
    return out


def cost_sweep(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, scheme: str = "equal",
               roundtrip_bps=(0, 1, 2, 5, 10)) -> pd.DataFrame:
    """Net Sharpe / CAGR of the combined book as the per-unit cost rises — where the blend dies."""
    w = combine(signals, panel, scheme=scheme)
    rows = {}
    for c in roundtrip_bps:
        s = summary(book_returns(w, panel, cost_bps=c))
        rows[c] = {"sharpe": s["sharpe"], "cagr": s["cagr"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out


def breakeven_cost_bps(signals: dict[str, pd.DataFrame], panel: pd.DataFrame, scheme: str = "equal",
                       hi: float = 50.0) -> float:
    """Per-unit cost (bps) at which the combined book's mean net return crosses zero: ``gross_mean /
    (turnover · 1e-4)``. Returns 0.0 if the book doesn't make money gross."""
    w = combine(signals, panel, scheme=scheme)
    gross_mean = book_returns(w, panel, cost_bps=0.0).mean()
    tpd = turnover(w)
    if tpd <= 0 or gross_mean <= 0:
        return 0.0
    return float(min(hi, gross_mean / (tpd * 1e-4)))


def correlation_matrix(signals: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    """Full pairwise correlation matrix of the components' standalone (gross) return streams."""
    rets = pd.DataFrame({nm: book_returns(w, panel, cost_bps=0.0) for nm, w in signals.items()}).dropna()
    return rets.corr()
