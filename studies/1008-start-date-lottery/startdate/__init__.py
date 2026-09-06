"""Study 1008 — The Start-Date Lottery.

An investor who began contributing in 1980 and one who began in 1999 followed
identical rules and ended up in different worlds. This is usually filed under "sequence risk"
and left there. It deserves better, because the size of the effect is measurable, its causes are
separable, and the standard remedies can be scored against each other rather than asserted.

The study asks three questions in order: how large is the lottery, what drives it, and which of
the things an investor actually controls — contribution schedule, glide path, diversification,
withdrawal flexibility — reduces it most per unit of expected return given up.

- :mod:`startdate.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`startdate.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
