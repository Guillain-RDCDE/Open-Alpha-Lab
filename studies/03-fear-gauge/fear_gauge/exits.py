"""Exit rules: once the fear gauge has put us long, when do we let go?

Identical in spirit to Study 02 — the gauge only decides *entry*; the exit is a
property of the *market* position, so the machinery is shared. Three primitives,
combined first-touch:

    * time   — hold a fixed number of trading days, then sell at the close.
    * target — sell when the S&P rebounds to ``entry * (1 + target)``.
    * stop   — sell when it falls to ``entry * (1 - stop)``. Without a stop, "buy
      the panic" quietly becomes "buy and pray" — which is the martingale beat 6
      warns about.

Intrabar convention (deliberately conservative): if one day's range touches *both*
the stop and the target, the **stop** is assumed to fill first. Buying into a VIX
spike should never be flattered by optimistic fills.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitRule:
    """A first-touch exit policy.

    Args:
        max_hold: maximum holding period in trading days (the time exit), >= 1.
        target:   take-profit as a positive fraction (``0.02`` = +2%); ``None`` off.
        stop:     stop-loss as a positive fraction (``0.05`` = -5%); ``None`` off.
    """

    max_hold: int = 5
    target: float | None = None
    stop: float | None = None

    def label(self) -> str:
        bits = [f"hold<={self.max_hold}d"]
        if self.target is not None:
            bits.append(f"tp+{self.target:.0%}")
        if self.stop is not None:
            bits.append(f"sl-{self.stop:.0%}")
        return " ".join(bits)


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    holding_days: int
    ret: float
    reason: str  # 'target' | 'stop' | 'time'


def resolve_trade(
    market: pd.DataFrame,
    entry_pos: int,
    entry_price: float,
    rule: ExitRule,
) -> Trade | None:
    """Walk a single trade forward from ``entry_pos`` under ``rule``.

    ``entry_pos`` is the integer position of the bar on which we are already long at
    ``entry_price`` (the trigger day's close). We scan subsequent bars' High/Low for
    target/stop touches, falling back to the time exit at the ``max_hold``-th close.
    Returns ``None`` if there is no bar after entry (end of sample).
    """
    n = len(market)
    if entry_pos + 1 >= n:
        return None

    highs = market["High"].to_numpy()
    lows = market["Low"].to_numpy()
    closes = market["Close"].to_numpy()
    dates = market.index

    target_price = entry_price * (1.0 + rule.target) if rule.target is not None else None
    stop_price = entry_price * (1.0 - rule.stop) if rule.stop is not None else None

    last = min(entry_pos + rule.max_hold, n - 1)
    for pos in range(entry_pos + 1, last + 1):
        if stop_price is not None and lows[pos] <= stop_price:
            return _make_trade(dates, entry_pos, pos, entry_price, stop_price, "stop")
        if target_price is not None and highs[pos] >= target_price:
            return _make_trade(dates, entry_pos, pos, entry_price, target_price, "target")

    return _make_trade(dates, entry_pos, last, entry_price, closes[last], "time")


def _make_trade(dates, entry_pos, exit_pos, entry_price, exit_price, reason) -> Trade:
    return Trade(
        entry_date=dates[entry_pos],
        exit_date=dates[exit_pos],
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        holding_days=int(exit_pos - entry_pos),
        ret=float(exit_price / entry_price - 1.0),
        reason=reason,
    )


def default_grid() -> list[ExitRule]:
    """A default sweep of exit rules for the family scan.

    Mixes pure-time exits with target/stop combinations so the scan can show the
    trade-off between letting the rebound run and cutting a gauge that keeps rising
    (i.e. a market that keeps falling).
    """
    grid: list[ExitRule] = []
    for h in (1, 5, 10, 21):
        grid.append(ExitRule(max_hold=h))                       # pure time
    for h in (10, 21, 42):
        grid.append(ExitRule(max_hold=h, target=0.03, stop=0.05))
        grid.append(ExitRule(max_hold=h, target=0.05, stop=0.07))
        grid.append(ExitRule(max_hold=h, target=0.05, stop=0.10))
    return grid
