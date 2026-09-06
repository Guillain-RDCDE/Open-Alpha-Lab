"""Study 996 — The Palindrome Portfolio.

This study looks for a pattern that cannot exist. Palindromic dates — 22/02/2022,
12/02/2021 — have no conceivable mechanism: no earnings land on them, no fund rebalances to
them, no human behaviour attaches to them. Any effect found is therefore, with certainty, a
false positive. That certainty is what makes them useful: they are a **calibrated ruler for
data mining**. Whatever a search over palindromes turns up is exactly what the same search would
turn up on a real hypothesis that happens to be false — and the number is much larger than most
people expect.

- :mod:`palindrome.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`palindrome.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
