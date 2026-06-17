"""Study 242 — Quality-Minus-Junk: does Asness's QMJ factor survive outside the backtest?

Asness, Frazzini & Pedersen (2019) "Quality Minus Junk": a composite quality score
(profitability + growth + safety) built from EDGAR fundamentals predicts the cross-section.
Long quality / short junk, annually rebalanced.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
