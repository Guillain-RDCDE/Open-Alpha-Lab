"""Study 969 — Log or Simple.

Log returns aggregate across *time* by addition; simple returns aggregate across
*assets* by weighting. You cannot have both, every codebase mixes them somewhere, and the size
of the resulting error is almost never measured. This study measures it — on tapes from T-bills
to bitcoin — and states the rule that settles each case.

- :mod:`log_vs_simple.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`log_vs_simple.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
