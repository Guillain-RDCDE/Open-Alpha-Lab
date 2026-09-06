"""Study 991 — The Slow Bell.

The central limit theorem says that a sum of many independent draws tends to a
normal distribution. Monthly returns are sums of about 21 daily returns; annual returns are sums
of about 252. So annual returns should be nearly normal — and the textbooks that use normal
arithmetic for long-horizon planning lean on exactly that. This study measures how fast the
convergence actually happens, and finds that the two conditions the theorem needs — independence
and finite variance — are both violated in ways that slow it down by more than most people
expect.

- :mod:`slowbell.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`slowbell.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
