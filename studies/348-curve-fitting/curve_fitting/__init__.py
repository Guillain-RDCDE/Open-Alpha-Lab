"""Study 348 — Curve-Fitting: the parameter-optimisation mirage.

A research-method demo, not a strategy. Tune a moving-average crossover by sweeping its
two windows over an *in-sample* slice, crown the best pair, then run that winner *out of
sample* and watch the headline collapse toward noise. The point is the gap between the
optimised in-sample Sharpe and the honest out-of-sample Sharpe — and how that gap grows
with the size of the parameter grid (the more dials you twist, the louder the mirage).

Distinct from the desk's many *single-rule* MA-crossover tests (Faber timing, golden/death
cross, Supertrend, …): those ask "does THIS rule work?"; this study asks the meta-question
"does *picking the best* rule on past data tell you anything about the future?" — the
answer the whole bench leans on every time it refuses to grid-search a verdict.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
