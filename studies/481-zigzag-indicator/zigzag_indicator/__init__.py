"""Study 481 — ZigZag Indicator (swing filter).

A mechanical, falsifiable encoding of the classic ZigZag indicator: it connects
alternating swing highs and lows whenever price reverses by more than an ``x%``
threshold, filtering out the noise between swings. The folklore says the ZigZag
"identifies turns" — a confirmed up-leg marks a tradable low. The catch is that
the ZigZag **repaints**: the most recent leg is provisional and only finalised
once price has moved ``x%`` the other way, so any naive "trade the last pivot"
backtest peeks at the future. We test only **confirmed** legs (long on a confirmed
up-leg, entered the next close) against random-entry and shuffled-leg placebos,
with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
