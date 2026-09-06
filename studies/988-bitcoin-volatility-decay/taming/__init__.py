"""Study 988 — The Taming.

"Bitcoin is maturing" is said at the top of every cycle and at the bottom of every
one. The claim is usually supported by a chart of realised volatility that slopes down over the
speaker's chosen window. This study asks whether the slope survives the obvious objections: that
volatility is enormously persistent, so a downward-sloping window is the *expected* appearance
of a mean-reverting process observed from a peak; that the sample contains three or four regimes
and a single trend line through them describes none; and that the standard error on a volatility
trend is much larger than anyone quotes.

- :mod:`taming.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`taming.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
