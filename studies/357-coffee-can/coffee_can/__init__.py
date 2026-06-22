"""Study 357 — the "Coffee Can" portfolio (buy great names, never sell).

Robert Kirby's parable: hand a client a basket of quality stocks, put the certificates in
a coffee can, and forbid yourself from ever trading them for a decade. The buy-and-hold
"can" reliably beat the actively-managed account. The desk's question is the honest one:
*is the edge real, or is it survivorship?*

We test a zero-turnover buy-and-hold of a quality large-cap basket against (a) a periodically
rebalanced version of the SAME basket and (b) the index (SPY), on real monthly total-return
data. The decomposition isolates two ingredients the parable fuses:

* **Low cost / low turnover** — a real, small, *defensible* tailwind (you don't pay the
  spread, the commission or the tax 20 times a year). This part is honest.
* **Survivorship** — the basket is chosen *because* its names are famous winners today. The
  same selection rule run on a panel that still contains the dead names (the honest universe)
  loses most of the edge. This part is a look-ahead artefact, named on the Signal axis.

See :mod:`coffee_can.data` (the real yfinance basket + a deterministic synthetic survivorship
generator as the positive control) and :mod:`coffee_can.strategy` (the buy-and-hold engine,
the paired t-test of the can vs the index, the cost model on NAV, and the survivorship
decomposition)."""

from . import data, strategy

__all__ = ["data", "strategy"]
