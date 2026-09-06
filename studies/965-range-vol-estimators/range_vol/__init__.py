"""Study 965 — The Range Estimators.

Parkinson (1980), Garman-Klass (1980), Rogers-Satchell (1991) and Yang-Zhang (2000)
estimate a day's variance from its high, low and open instead of from its close alone, and
the textbooks quote efficiency gains of five to eight times. This study checks that claim on
a simulation where the answer is known, then asks the question the textbooks skip: what
happens to those estimators on a market that **gaps overnight**, and does any of it improve a
forecast you would actually use?

- :mod:`range_vol.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`range_vol.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
