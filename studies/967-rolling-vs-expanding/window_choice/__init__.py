"""Study 967 — Window Shopping.

Every number a portfolio uses — a beta, an expected return, a covariance matrix — is
estimated from a window of history, and the window is almost always chosen by habit: five
years because that is what Bloomberg shows, or everything because more data must be better.
This study measures the choice out of sample on three different quantities, and finds that
they do not want the same answer.

- :mod:`window_choice.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`window_choice.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
