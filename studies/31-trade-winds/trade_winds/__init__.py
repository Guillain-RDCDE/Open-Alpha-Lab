"""Trade-Winds — cross-asset time-series momentum (trend-following / managed futures).

Unlike *cross-sectional* equity momentum (buy past winners vs losers — fragile, crash-prone),
*time-series* momentum asks a simpler question of EACH market on its own: has it been going up or
down lately? Go long what's been rising, short what's been falling, size every market to the same
risk, and hold a diversified basket across equities, bonds, commodities and FX. It is the engine of
the managed-futures industry, and its signature is **crisis alpha** — it tends to *make* money when
stocks crash, because crashes are trends.

Modules:
  * :mod:`trade_winds.data`      — offline synthetic trending panel (+ a random-walk null); cached real futures.
  * :mod:`trade_winds.strategy`  — the TSMOM signal, per-market vol-scaling, equal-risk portfolio, target vol.
  * :mod:`trade_winds.costs`     — transaction costs and the net book (futures are cheap — that matters).
  * :mod:`trade_winds.extension` — beat-7: the lookback sweep and the crisis-alpha measurement.
"""

from . import costs, data, extension, strategy  # noqa: F401

__all__ = ["data", "strategy", "costs", "extension"]
