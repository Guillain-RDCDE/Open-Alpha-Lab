"""Study 989 — The One-Way Beta.

The pitch for an altcoin is leverage on Bitcoin: when Bitcoin rises 10% the altcoin
rises 20%, so you get the same view with more of it. The pitch is never stated as a *symmetric*
claim, but it is always heard as one. This study measures the up-beta and the down-beta
separately — and spends most of its effort on the reason a difference between them is so easy
to manufacture out of nothing.

- :mod:`onewaybeta.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`onewaybeta.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
