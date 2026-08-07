"""Study 835 — Spurious Regression (Granger & Newbold 1974).

Regress one **independent random walk** on another and OLS hands you a large *t*-stat and
a high R² — a "significant" relation that does not exist, manufactured purely by the
nonstationarity of the two series. We simulate many such pairs, show the level-OLS *t* is
grossly over-sized (and gets *worse* with more data), and demonstrate the fixes:
first-differencing restores the correct 5% size, and a cointegration test tells a genuine
long-run relation from a spurious one.

* ``data``     — deterministic seeded generators: two *independent* random walks (the
                 pitfall world), two *independent stationary* series (the specificity
                 control), and a genuinely *cointegrated* pair (the positive control).
* ``strategy`` — the vectorised batch OLS, the level-vs-difference experiment, the
                 sample-size sweep, the Engle-Granger cointegration leg, a costed pairs
                 timer, and the inference primitives (one-sample / Welch / Newey-West /
                 Wilson) shared with the desk's canonical template.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
