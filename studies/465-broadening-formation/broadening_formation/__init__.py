"""Study 465 — Broadening Formation (megaphone top).

A mechanical, falsifiable encoding of the chart-pattern folklore that a *broadening
formation* — diverging swing highs (higher highs) and swing lows (lower lows), an
expanding "megaphone" range — marks an exhausted, reversing top. We detect the megaphone
mechanically from confirmed swing pivots, fire a **short on the break of the lower
boundary** (entered at the next close), and test the forward 5/10/20/60-day return against
a drift-matched random-entry baseline, a geometry placebo, and a synthetic planted control.

The third-axis question: *does expanding volatility forecast a turn?*
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
