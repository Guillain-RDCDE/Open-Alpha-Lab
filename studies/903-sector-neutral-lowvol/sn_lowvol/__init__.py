"""Study 903 — Sector-Neutral Low-Vol.

Does the low-volatility anomaly survive once its defensive-**sector** bet is stripped out?
Rank each name on trailing volatility *within its own sector* (demean by the sector median),
then long the low-vol / short the high-vol names sector-neutrally.

``data``     — real cross-section loader (survivorship-guarded) + sector map + synthetic control.
``strategy`` — trailing-vol signal, sector demean, the sort, inference primitives, timer.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
