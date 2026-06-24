"""Study 445 — Elliott Wave (the tightest mechanical proxy a proponent would accept).

Elliott Wave Theory is irreducibly *subjective* — two analysts label the same chart
differently, and the count is only "confirmed" after the fact. So there is no single
falsifiable rule. What we test is the **tightest mechanical version proponents do accept**:
a standard ZigZag swing detector to mark the pivots, a Fibonacci filter on the wave-2
retracement, and the one *forward-looking, tradable* prediction the theory makes — after a
clean impulse 1-2 with a textbook wave-2 retracement, **wave 3 extends in the impulse
direction** (the entry every Elliottician wants), and after a completed five-wave impulse a
correction follows. t-stat, label-shuffle placebo, costs.
"""

from . import data, strategy  # noqa: F401
