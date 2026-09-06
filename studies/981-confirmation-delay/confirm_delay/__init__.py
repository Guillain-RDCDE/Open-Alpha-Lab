"""Study 981 — The Price of Waiting.

"Wait for confirmation" is the most universally repeated piece of trading discipline
there is, and one of the least measured. Requiring a signal to persist for *k* days before
acting is a real trade-off: fewer false starts, later entries and exits. This study prices both
sides on three standard signals, four tapes and a whipsaw metric that separates the two
effects.

- :mod:`confirm_delay.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`confirm_delay.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
