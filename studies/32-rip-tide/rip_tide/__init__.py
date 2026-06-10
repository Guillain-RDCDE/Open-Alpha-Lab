"""Study 32 — Rip-Tide: short-horizon contrarian (mean-reversion) trading on liquid futures.

The mirror image of Study 31 (Trade-Winds / time-series momentum): instead of *riding* the recent
move, **fade** it — sell the markets that just rose, buy the ones that just fell, on a 1–5 day horizon.
The book is built with the *same* equal-risk, vol-targeted machinery as the trend book, so the only
thing that changes is the sign and the speed — which is exactly what makes the cost story brutal.
"""

from . import costs, data, extension, strategy  # noqa: F401
