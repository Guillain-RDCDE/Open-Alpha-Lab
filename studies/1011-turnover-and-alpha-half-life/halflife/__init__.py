"""Study 1011 — The Half-Life of an Edge.

Two facts about a trading signal are usually reported: how strong it is, and how
much it returned in a backtest. A third decides whether either matters — **how fast it decays**.
A signal whose information coefficient halves in three days and one whose IC halves in three
months are different businesses with different capacity, different cost tolerance and different
optimal portfolios, even if their headline ICs are identical.

Grinold's fundamental law connects the three: information ratio ≈ IC × √breadth, where breadth
is the number of independent bets per year — which is set by the decay rate. Gârleanu and
Pedersen (2013) then show what to do when trading is costly: trade partway toward the target,
at a rate determined by the decay. This study measures decay directly, tests both results, and
finds where the theory's assumptions bite.

- :mod:`halflife.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`halflife.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
