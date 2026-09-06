"""Study 1012 — Choose Your Benchmark.

Alpha is defined as what is left over after a benchmark is subtracted. That makes
it a statement about two things — the manager and the benchmark — and only one of them is
usually discussed. This study measures how much of a reported alpha is a property of the choice,
whether the data can settle that choice, and what a sceptical reader should ask for.

The uncomfortable finding is not that the benchmark matters. It is that the range of *defensible*
benchmarks is wide enough to move the answer past the point of decision, and that the standard
statistical tools for choosing one are far weaker than the confidence with which alphas are
reported.

- :mod:`benchmark.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`benchmark.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
