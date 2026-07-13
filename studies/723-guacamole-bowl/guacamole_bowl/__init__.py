"""Study 723 — Guacamole-Bowl: does the Super-Bowl guacamole binge print a Jan–Feb seasonal?

The folklore: America eats a mountain of guacamole for the Super Bowl (early February), so the
avocado / produce trade should carry a January–February seasonal — buy ahead of the game, ride the
surge. We test the strongest tradable version on ``PEP`` (PepsiCo/Frito-Lay: Tostitos + the branded
dips, the Super-Bowl chip-and-dip complex — a *labelled proxy*, because the pure-play avocado name
Calavo (CVGW) is unavailable on the current Yahoo feed), benchmarked against ``SPY``: per-month HAC
t-stats, a Jan–Feb window spread, a placebo across all 66 month-pairs, a block-bootstrap CI, a Jan–Feb
timer vs buy-and-hold, and a Newey-West alpha. A cited, approximate wholesale-Hass seasonal index
shows the avocado price's own (awkward-for-the-folklore) shape but never backs a Signal stamp.

The offline control is a synthetic world with a tunable Jan–Feb premium and a null — it pins the
machinery; it can never back a Signal stamp (METHODOLOGY → the inference bar).
"""

from . import data, strategy  # noqa: F401
