"""Study 993 — Down Hurts More.

Black noticed it in 1976: stock volatility rises more after a fall than after an
equally-sized rise. He proposed a mechanism — a falling price raises the debt-to-equity ratio,
making the remaining equity riskier — and the name "leverage effect" stuck. The effect is one of
the most robust findings in empirical finance. The explanation has been in serious doubt for
forty years, and the cleanest way to see why is to run the same test on assets that have no
debt at all.

- :mod:`downhurts.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`downhurts.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
