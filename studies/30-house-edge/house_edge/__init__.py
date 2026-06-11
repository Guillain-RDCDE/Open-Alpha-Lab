"""House-Edge — the true cost of leverage in a systematic timing strategy.

A vol-targeted contrarian dip-buyer with a macro gate genuinely *cuts risk* — and funded at
the fair bill rate it nearly matches the index. But it never *beats* it, and the retail
pitch dies at the broker's financing **markup**: a CFD charges that markup on the whole
notional (not just the borrowed slice), the broker's "house edge". The drawdown protection
is real; the market-beating is a mirage, and the mirage is the markup.

Modules:
  * :mod:`house_edge.data`      — offline synthetic price path (vol-clustered + crashes); cached real fetch.
  * :mod:`house_edge.strategy`  — the contrarian dip-buyer + vol-targeting + macro gate; exposure & metrics.
  * :mod:`house_edge.costs`     — the heart: margin vs futures/CFD financing, and where the markup lands.
  * :mod:`house_edge.extension` — beat-7 worked complement: the markup sweep where the dream dies.
"""

from . import costs, data, extension, strategy  # noqa: F401

__all__ = ["data", "strategy", "costs", "extension"]
