"""Study 978 — The Resampled Frontier.

Richard Michaud's resampled efficiency answers estimation error by embracing it: draw
bootstrap samples of the return history, optimise on each, and average the resulting weight
vectors. The portfolios come out smoother and more diversified, and the method has been
defended and attacked for thirty years. This study runs it out of sample against the two
things it competes with — shrinkage and simply not optimising — and asks whether the averaging
is doing anything the cheap fixes do not.

- :mod:`resampled.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`resampled.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
