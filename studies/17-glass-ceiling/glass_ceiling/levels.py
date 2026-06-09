"""Where the trade lives: the resistance the breakout clears and the swing low the stop hides under.

Koroush AK's setup needs three geometric primitives off the bar tape, and every one is a *decision*
about how to mechanize a chart a trader draws by eye. We state each choice so the backtest is
reproducible and the steelman is honest:

    * **Resistance** = the highest prior *close* over a trailing ``lookback`` window. Using the close
      (not the intrabar high) matches the entry rule, which is defined on closes; it is the cleanest
      mechanical stand-in for the hand-drawn level and keeps the generator's breakout definition and
      the strategy's trigger on the same footing.
    * **Breakout trigger** = ``confirm`` consecutive closes strictly above that level (Koroush's "two
      1-minute candle closes"). Confirmation is what separates a momentum entry from chasing a single
      spike bar.
    * **Swing low** = the lowest low since the run-up began — specifically the minimum ``Low`` over a
      trailing window back to the level's origin — which is where the stop sits. Koroush floors the
      stop distance at 1% of price ("if it's less than 1%, use the next swing low down"); we expose
      that floor as ``min_stop_frac`` so the floored and unfloored versions are both testable.

These are deliberately *generous* (steelman) readings: a clean rolling high, confirmation before
entry, a real swing-low stop. If the edge dies even on this charitable mechanization, it is the
idea that fails, not a strawman of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resistance(bars: pd.DataFrame, lookback: int = 30) -> pd.Series:
    """Trailing resistance: the max prior **close** over ``lookback`` bars (excluding the current bar).

    Shifted by one so the level at bar ``t`` uses only closes up to ``t-1`` — no look-ahead. The
    first ``lookback`` bars are ``NaN`` (no level yet).
    """
    return bars["Close"].rolling(lookback).max().shift(1).rename("resistance")


def breakout_triggers(bars: pd.DataFrame, lookback: int = 30, confirm: int = 2) -> pd.Series:
    """Boolean series: ``True`` at bar ``t`` when the last ``confirm`` closes are all above resistance.

    The level compared against is the resistance *as of the start of the confirmation run* (the level
    at the first of the ``confirm`` bars), so a rising rolling-max can't invalidate a break the moment
    price clears it. A trigger marks a *candidate* entry; :mod:`strategy` then enforces one position
    at a time.
    """
    level = resistance(bars, lookback)
    close = bars["Close"]
    # level frozen at the first bar of the confirmation window:
    level_at_run_start = level.shift(confirm - 1)
    above = close > level_at_run_start
    trig = above.copy()
    for k in range(1, confirm):
        trig &= above.shift(k).fillna(False)
    return (trig & level_at_run_start.notna()).rename("trigger")


def swing_low(bars: pd.DataFrame, idx: int, window: int = 30) -> float:
    """Lowest ``Low`` over the ``window`` bars ending at ``idx`` — the stop's hiding place."""
    lo = max(0, idx - window + 1)
    return float(bars["Low"].iloc[lo: idx + 1].min())


def stop_from_swing(
    entry_price: float,
    swing: float,
    min_stop_frac: float = 0.01,
    bars: pd.DataFrame | None = None,
    idx: int | None = None,
    window: int = 30,
) -> float:
    """The stop price: the swing low, but pushed down to honour Koroush's ``min_stop_frac`` floor.

    If the nearest swing low sits closer than ``min_stop_frac`` of price, the rule says step to the
    *next* swing low down. When ``bars``/``idx`` are given we look for a genuinely lower low further
    back; otherwise we fall back to placing the stop exactly ``min_stop_frac`` below entry (the
    floor). Either way the returned stop is never closer than the floor — which is the load-bearing
    point, because a too-tight stop is what manufactures fake stop-outs.
    """
    stop = swing
    floor_price = entry_price * (1.0 - min_stop_frac)
    if stop >= floor_price and bars is not None and idx is not None:
        # walk further back for the next lower swing low that clears the floor
        lo = max(0, idx - 5 * window)
        lows = bars["Low"].iloc[lo: idx + 1]
        candidates = lows[lows < floor_price]
        stop = float(candidates.max()) if not candidates.empty else floor_price
    elif stop >= floor_price:
        stop = floor_price
    return float(min(stop, floor_price))
