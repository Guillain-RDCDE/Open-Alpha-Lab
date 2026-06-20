"""Study 337 — Covered-Call-ETF: the buy-write *income illusion* (JEPI/QYLD) vs SPY total return.

The distinct angle vs Study 62 (Premium-Seller, QYLD-vs-its-own-underlying capture): here the
object is the **income illusion** — distribution yield decomposed against NAV erosion — across the
*newer* JEPI/JEPQ generation as well as QYLD, raced against **SPY total return**, with a mechanical
synthetic **buy-write replicator** as the offline control.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
