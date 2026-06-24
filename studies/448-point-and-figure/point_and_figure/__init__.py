"""Study 448 — Point & Figure price targets.

The folklore (Cohen, *How to Use the Three-Point Reversal Method of Point and Figure
Stock Market Trading*; du Plessis, *The Definitive Guide to Point and Figure*): a P&F
chart strips out time, plotting only meaningful price moves as columns of Xs (up) and
Os (down). When a column of Xs rises one box above the prior X-column you get a
**double-top buy**; the width of the preceding congestion (the **horizontal count**)
times the box size times the reversal gives a mechanical **price target** — and the
claim is that those targets get **hit**.

We encode the tightest *objective* P&F a practitioner would accept — a fixed
box size (a fraction of median price), a 3-box reversal, the standard double-top /
double-bottom signals, and the horizontal-count target — and ask one falsifiable
question: **does price reach the count target before it stops out**, more often than a
fair geometric baseline (a target placed the same distance away in a flat-drift world)?

See :mod:`point_and_figure.data` (real cache-first loader + deterministic synthetic
control with a planted "targets-get-hit" knob) and :mod:`point_and_figure.strategy`
(the P&F column engine, the double-top/-bottom signals, the horizontal-count target,
the hit-rate arbiters, a same-distance random-target placebo, and costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
