"""Study 724 — Pumpkin-Spice-Season: does the PSL launch buy you a tradable autumn in SBUX?

The folklore: Starbucks launches the Pumpkin Spice Latte in late August, kicking off "pumpkin
spice season" — a cultural juggernaut that (the story goes) lights up Starbucks' autumn quarter and
sends the stock *beating the market* from August into November. We test the tradable version of that
claim on **SBUX total-return excess over SPY** (1993→2026): per-month one-sample HAC t-stats on the
excess series, a season (Aug–Nov) vs off-season Welch spread, a block-bootstrap CI on the headline, a
12-window placebo that asks whether Aug–Nov is special among all four-month windows, and a seasonal
rotation raced against buy-and-hold — gross and net of costs. A QSR-basket leg (SBUX/MCD/YUM/CMG)
checks robustness.

The offline control is a synthetic world with a tunable PSL premium and a null — it pins the
machinery; it can never back a Signal stamp (METHODOLOGY → the inference bar).
"""

from . import data, strategy  # noqa: F401
