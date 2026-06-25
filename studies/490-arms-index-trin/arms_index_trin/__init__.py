"""Study 490 — Arms Index (TRIN), a market-breadth timing rule.

Richard Arms' *Trading Index* (1967):

    TRIN = (advancing issues / declining issues) / (advancing volume / declining volume).

The folklore says a **high** TRIN marks a panic/washout — heavy volume piling into the few
decliners — and therefore precedes a short-term **bounce**, while a low TRIN marks euphoria.
We encode that as a forward-return study: build a breadth proxy from a small basket of liquid
ETFs (true exchange breadth is unavailable offline), fire a "buy after a high-TRIN panic day"
rule, and race it against a drift-matched random-entry baseline plus a shuffled-TRIN placebo,
with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
