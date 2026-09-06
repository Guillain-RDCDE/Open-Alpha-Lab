"""Study 979 — The Prior Is the Portfolio.

Black-Litterman is famous for producing sensible portfolios where mean-variance
optimisation produces absurd ones. Its least-discussed property is that with **no views** it
returns the prior portfolio exactly — so everything it does is either the prior or the views,
and the model itself contributes only the arithmetic that mixes them. This study verifies the
identity, calibrates how strong a view must be before the portfolio moves, and prices a
view-driven tilt out of sample.

- :mod:`bl_prior.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`bl_prior.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
