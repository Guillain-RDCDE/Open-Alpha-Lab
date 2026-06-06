"""Exit rules: once we've caught the knife, when do we let go?

We model three primitive exits and let them combine:

    * time   — hold a fixed number of trading days, then sell at the close.
    * target — sell when price rebounds to ``entry * (1 + target)``.
    * stop   — sell when price falls to ``entry * (1 - stop)`` (the knife kept
      falling). Without a stop, a "buy the dip" strategy quietly turns into
      "buy and pray" during a real bear market.

The hybrid rule is *first-touch*: whichever of stop / target / time happens
first ends the trade. These are kept as parameters, not hard-coded, precisely so
the analysis can *sweep* them and show which (if any) survive — rather than us
quietly picking the one that looks best.

Intrabar convention (deliberately conservative): if a single day's range touches
*both* the stop and the target, we assume the **stop** filled first. Catching a
falling knife should never be flattered by optimistic fill assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitRule:
    """A first-touch exit policy.

    Args:
        max_hold:   maximum holding period in trading days (the time exit). Must
                    be >= 1.
        target:     take-profit as a positive fraction (e.g. ``0.02`` = +2%).
                    ``None`` disables it.
        stop:       stop-loss as a positive fraction (e.g. ``0.05`` = -5% from
                    entry). ``None`` disables it.
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
    ohlc: pd.DataFrame,
    entry_pos: int,
    entry_price: float,
    rule: ExitRule,
) -> Trade | None:
    """Walk a single trade forward from ``entry_pos`` under ``rule``.

    ``entry_pos`` is the *integer* position in ``ohlc`` of the bar on which we are
    already long at ``entry_price`` (typically the trigger day's close). We then
    scan subsequent bars' High/Low for target/stop touches, and fall back to the
    time exit at the close of the ``max_hold``-th bar.

    Returns ``None`` if there is not even one bar after entry (end of sample).
    """
    n = len(ohlc)
    if entry_pos + 1 >= n:
        return None

    highs = ohlc["High"].to_numpy()
    lows = ohlc["Low"].to_numpy()
    closes = ohlc["Close"].to_numpy()
    dates = ohlc.index

    target_price = entry_price * (1.0 + rule.target) if rule.target is not None else None
    stop_price = entry_price * (1.0 - rule.stop) if rule.stop is not None else None

    last = min(entry_pos + rule.max_hold, n - 1)
    for pos in range(entry_pos + 1, last + 1):
        # Conservative ordering: check the stop before the target.
        if stop_price is not None and lows[pos] <= stop_price:
            return _make_trade(dates, entry_pos, pos, entry_price, stop_price, "stop")
        if target_price is not None and highs[pos] >= target_price:
            return _make_trade(dates, entry_pos, pos, entry_price, target_price, "target")

    # Time exit at the close of the final eligible bar.
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
    """A sensible default sweep of exit rules for the family scan.

    Mixes pure-time exits (no target/stop) with a few target/stop combinations
    so the analysis can show the trade-off between letting winners run and
    cutting falling knives.
    """
    grid: list[ExitRule] = []
    for h in (1, 3, 5, 10, 20):
        grid.append(ExitRule(max_hold=h))                       # pure time
    for h in (5, 10, 20):
        grid.append(ExitRule(max_hold=h, target=0.02, stop=0.05))
        grid.append(ExitRule(max_hold=h, target=0.05, stop=0.05))
        grid.append(ExitRule(max_hold=h, target=0.03, stop=0.10))
    return grid
