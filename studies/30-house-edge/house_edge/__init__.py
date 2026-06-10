"""House-Edge — the true cost of leverage in a systematic timing strategy.

A vol-targeted contrarian dip-buyer with a macro gate genuinely *cuts risk* — but once
you charge leverage its real, full-notional financing cost (the broker's "house edge"),
the apparent return advantage over a plain total-return index vanishes. The drawdown
protection is real; the return edge is an accounting illusion.

Modules:
  * :mod:`house_edge.data`      — offline synthetic price path (vol-clustered + crashes); cached real fetch.
  * :mod:`house_edge.strategy`  — the contrarian dip-buyer + vol-targeting + macro gate; exposure & metrics.
  * :mod:`house_edge.costs`     — the heart: idealized vs HONEST (full-notional) CFD/leverage financing.
  * :mod:`house_edge.extension` — beat-7 worked complement: the financing-rate sweep where the edge dies.
"""

from . import costs, data, extension, strategy  # noqa: F401

__all__ = ["data", "strategy", "costs", "extension"]
