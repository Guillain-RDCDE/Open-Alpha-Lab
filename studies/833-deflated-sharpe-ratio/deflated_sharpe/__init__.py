"""Study 833 — the Deflated Sharpe Ratio.

Bailey & López de Prado (2014): run ``N`` independent strategies on a tape with **zero** true
edge and the *best* sample Sharpe is not zero — it inflates with ``N`` per the
expected-maximum-Sharpe formula. The **Deflated Sharpe Ratio** re-scores that winner against
the expected maximum under the null and correctly shrinks it to a coin flip, while sparing an
honestly-good single strategy.

* ``data``     — the deterministic seeded NULL panel (N independent, true-zero-edge columns)
                 and the honest positive control (a single genuine-edge stream). No network.
* ``strategy`` — the expected-maximum-Sharpe formula, the inflation curve, the Deflated /
                 Probabilistic Sharpe Ratio, the IS→OOS champion collapse, the costed timer,
                 the shared inference rails, and the null / honest synthetic controls.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
