"""Study 823 — Variance-Risk-Premium Return Predictor.

Bollerslev, Tauchen & Zhou (2009): the **variance risk premium** — implied variance
(from VIX) minus realized variance (from the underlying) — **predicts the aggregate
market's forward excess return** in a time-series predictive regression, with the R²
peaking near the quarterly horizon. Positive VRP → higher forward return.

* ``data``     — the real SPY + ^VIX daily tape (yfinance, cached under this study's own
                 ``_cache/``) plus a deterministic seeded synthetic positive control
                 (a planted VRP→return predictive relation, null at ``edge=0``).
* ``strategy`` — the monthly IV/RV/VRP build, the forward-return construction, the
                 predictive regression with a Newey-West slope *t*, a block-bootstrap
                 placebo, a horizon/era robustness sweep, a costed timer, and the
                 seed-robust synthetic control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
