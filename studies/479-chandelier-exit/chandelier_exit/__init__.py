"""Study 479 — Chandelier Exit (ATR trailing stop).

Chuck LeBeau's chandelier exit hangs a long-flat trailing stop a fixed multiple of the
Average True Range below the highest high since entry: ``stop = HH(n) - m * ATR(n)`` with
the canonical ``n = 22``, ``m = 3``. The folklore is that this volatility-scaled trailing
stop "lets winners run and cuts losers", beating a passive buy-and-hold. We test that as a
position/forward-return study against buy-and-hold and a drift-matched random-stop baseline,
with costs, plus a placebo that scrambles the trailing-stop geometry while keeping its
marginal so we can ask whether the *ATR trail itself* is load-bearing.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
