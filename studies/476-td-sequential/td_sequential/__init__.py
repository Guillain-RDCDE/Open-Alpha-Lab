"""Study 476 — TD Sequential (DeMark 9-13 exhaustion count).

A mechanical, falsifiable encoding of Tom DeMark's TD Sequential: a **TD Buy Setup**
is nine consecutive closes each below the close four bars earlier; a completed setup
arms a **TD Buy Countdown** that increments (up to 13) when the close is at/below the
low two bars earlier. The folklore says a completed buy setup (9) — and especially a
completed countdown (13) — marks **exhaustion** of the down-move and a high-probability
long. We test that as a forward-return study against a drift-matched random-entry
baseline and a phase-scrambled placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
