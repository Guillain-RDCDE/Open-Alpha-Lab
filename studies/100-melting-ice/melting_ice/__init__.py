"""Study 100 — Melting-Ice: the leveraged-ETF 'volatility decay' folklore, decomposed.

Is a 3x ETF (TQQQ, UPRO) really doomed to bleed to zero by 'volatility decay'? We
implement the exact constant-leverage rebalancing identity from the daily underlying
returns, split the gap-to-naive-3x into the (negative) variance-drag term and the
(positive) compounding-in-a-trend term, and let the real tape decide which wins.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
