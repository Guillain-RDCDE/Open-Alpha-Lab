"""Study 828 — FX Dollar Factor (DOL) and dollar-timing.

Lustig, Roussanov & Verdelhan (2011): the **dollar factor** ``DOL`` — the
equal-weight average excess return of a basket of foreign currencies against the USD
— is a candidate priced risk factor. Two questions:

1. **The DOL premium.** Does a long basket of foreign currencies funded in USD earn a
   positive average return (does the dollar systematically *depreciate*)?
2. **Dollar-timing.** Does the **average forward discount** (the mean interest-rate
   gap of the basket vs USD) *time* DOL — i.e. is next-month DOL higher when the
   conditioning signal is high?

* ``data``     — the real tape (yfinance daily FX spot for a fixed G10 basket vs USD,
                 cached under the study's own ``_cache/`` as a month-end parquet; every
                 pair normalised to **USD-per-foreign-currency** so a rise means the
                 foreign currency appreciated) plus a deterministic seeded synthetic
                 positive control (a planted common dollar drift + a planted timing
                 relation, null at ``edge=0`` / ``timing=0``).
* ``strategy`` — DOL construction, the premium test, the dollar-timing predictive
                 regression, the inference primitives (one-sample / Welch / Newey-West
                 HAC / Wilson / block-shuffle placebo), and the costed basket/timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
