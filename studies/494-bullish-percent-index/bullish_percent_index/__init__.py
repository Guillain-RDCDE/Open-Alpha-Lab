"""Study 494 — Bullish Percent Index (market-breadth oscillator).

A mechanical, falsifiable encoding of the Bullish Percent Index (BPI). True BPI is the
percentage of a basket whose stocks sit on a Point & Figure *buy* signal; we use the standard
desk proxy — the **percentage of the basket trading above its moving average** — to build a
0-100 breadth oscillator. The folklore says BPI *calls tops and bottoms*: BPI > 70 is an
"overbought" warning (sell SPY) and BPI < 30 an "oversold" green light (buy SPY). We test the
oversold-buy leg against a drift-matched random-entry baseline on SPY, plus a breadth-scramble
placebo and a synthetic planted-turn control, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
