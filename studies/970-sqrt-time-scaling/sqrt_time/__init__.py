"""Study 970 — Root Time.

"Annualise it by multiplying by the square root of 252." The formula is in every risk
system, every fact sheet and every interview question, and it is exactly right if and only if
returns are serially independent. This study measures the size of the error where they are not
— on ten tapes, at four horizons — using variance ratios, the Lo-MacKinlay test, and the
correction factor from Lo (2002).

- :mod:`sqrt_time.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`sqrt_time.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
