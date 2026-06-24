"""Study 430 — Klinger Oscillator.

Does Stephen Klinger's *Volume Force* oscillator — built to make long-term volume
"lead" price — actually time SPY better than buy-and-hold, or better than a plain
moving-average / MACD rule? The package exposes a cache-first data layer
(:mod:`klinger_oscillator.data`) and the detector + timing rules + honest arbiters
(:mod:`klinger_oscillator.strategy`).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
