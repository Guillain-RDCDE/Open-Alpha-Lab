"""Study 478 — Percentage Price Oscillator (PPO).

A mechanical, falsifiable encoding of the Percentage Price Oscillator — the
*normalized* cousin of MACD. PPO = 100*(EMA12 - EMA26)/EMA26, with a 9-period EMA
of the PPO as the *signal line*. The folklore says a **bullish crossover** (PPO
rising above its signal line) is a buy. We test that as a forward-return study
against a drift-matched random-entry baseline and a shuffled-crossover placebo,
with costs — and we ask the third-axis question: **does normalizing MACD add edge?**
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
