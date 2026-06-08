"""Coiled-Spring (Study 07) — the "20 EMA pivot breakout" swing rule, tested honestly.

The claim comes from a retail trading book (Jayesh Shah, *Trade the 20 EMA*): a stock
that breaks above its 20-period EMA, forms a pivot high, pulls back **without closing
below the EMA**, then breaks the pivot on **2x volume**, is supposed to spring into an
"explosive" +30-50% move in 6-10 days. The book sells the rule on a handful of hand-picked
winners — exactly the selection trap the desk exists to expose.

Modules:
  * :mod:`coiled_spring.data`       — cached real universe (split-only) + a synthetic one
                                       with **planted springboards** to recover offline.
  * :mod:`coiled_spring.signals`    — the mechanised rule: EMA, confirmed pivots, the
                                       hold-the-EMA pullback, the volume-gated breakout.
  * :mod:`coiled_spring.backtest`   — enter-next-open, stop under the breakout bar, trailing
                                       exit; the per-trade ledger and equity curve, with costs.
  * :mod:`coiled_spring.robustness` — the headline falsification (breakout vs same-stock
                                       random entry), bootstrap, decay-by-year, cost sweep.
"""

from . import backtest, data, robustness, signals

__all__ = ["data", "signals", "backtest", "robustness"]
