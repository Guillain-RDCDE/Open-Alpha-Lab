"""Study 307 — Coffee-Seasonality: does coffee have a tradable harvest/frost-season calendar?

Monthly seasonality of Arabica coffee futures (KC=F). The folklore: coffee is a Southern-Hemisphere
crop and Brazil grows ~a third of the world's beans, so the calendar should bite twice a year — the
**frost-risk window** (Jun–Aug, the Brazilian winter, when a single overnight frost can wipe out a
harvest and spike prices) is the bullish leg, and the **harvest-pressure window** (May–Sep, when new
crop floods the market) is the bearish leg. We test it on KC=F monthly data: per-month one-sample
HAC t-stats, a frost-vs-harvest spread, a block-bootstrap CI on the headline, and a long-frost /
short-harvest calendar timer raced against buy-and-hold (excess-of-cash Sharpe, both legs).

The offline control is a synthetic world with a tunable frost premium and a null — it pins the
machinery; it can never back a Signal stamp (METHODOLOGY → the inference bar).
"""

from . import data, strategy  # noqa: F401
