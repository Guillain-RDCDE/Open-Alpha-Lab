"""Study 471 — QQE (Quantitative Qualitative Estimation).

A mechanical, falsifiable encoding of the QQE indicator: a Wilder-smoothed RSI
(RSI MA) tracked by a *trailing band* whose width is an ATR-of-the-smoothed-RSI.
The folklore says a long fires when the smoothed RSI **crosses above** its QQE
trailing band — a momentum-ignition signal popular in the MetaTrader / TradingView
crowd. We test that as a forward-return study against a drift-matched random-entry
baseline and a phase-scramble placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
