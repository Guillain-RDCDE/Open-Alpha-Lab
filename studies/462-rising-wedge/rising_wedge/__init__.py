"""Study 462 — Rising Wedge (converging up-sloping support & resistance).

A mechanical, falsifiable encoding of the **rising wedge** chart pattern: two
*up-sloping* trendlines — support through swing lows, resistance through swing
highs — that **converge** (support rises faster, the channel narrows). Textbook
technical analysis calls it a **bearish** pattern: price is "supposed" to break
*down* through the lower line. The folklore says: short the lower-line break.

We encode that mechanically (confirmed-fractal pivots, both lines rising, both
positive-sloped, narrowing range), fire a SHORT on the first close below the
support line (entered at the next close, one lag), and test the forward return
against a drift-matched random-entry baseline and a slope-scramble placebo, with
costs. A short's "win" is a price *fall*, so the forward return is the **negative**
of the underlying move.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
