"""Study 908 — Optimized-Roll Commodities.

A front-month commodity index rolls naively up a rising futures curve and bleeds a
"contango tax"; an **optimized-roll** index (USCI — the SummerHaven Dynamic Commodity
Index, holding the 14 most-backwardated of 27 commodities in cheapest-carry contracts)
dodges much of it. We race the optimized wrapper against front-month rollers (GSG, DJP)
and the semi-optimized DB "Optimum Yield" index (DBC) on a **higher excess-of-cash
return / Sharpe** — every leg taken excess of cash (BIL), so the shared collateral-yield
component is netted out and only the spot + roll-yield difference remains.

* ``data``     — the real tape (yfinance total-return closes for USCI, DBC, PDBC, DJP,
                 GSG, BIL, cached under the study's own ``_cache/``) plus a deterministic
                 seeded synthetic world with a tunable planted roll edge (null at 0).
* ``strategy`` — the excess-of-cash frame, the excess-vs-excess Sharpe race (paired block
                 bootstrap on the advantage), HAC *t* on the return difference, era cut,
                 calendar-year table, and the costed version.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
