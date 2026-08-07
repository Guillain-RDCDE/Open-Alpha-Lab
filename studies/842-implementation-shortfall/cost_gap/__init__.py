"""Study 842 — Implementation Shortfall: the paper-vs-live cost gap.

A *research-method* demo, not a tradable signal. The claim we make undeniable: a
backtest without a cost model is meaningless. We take the SAME moderate-turnover
cross-sectional strategy — a signal we *planted* to carry a genuine gross edge, so
its paper (0-cost) Sharpe genuinely dazzles — and evaluate it across a ladder of
transaction costs (0 / realistic / stressed) with a turnover-scaled market-impact term
(~ participation). The gross alpha that looks Real on paper dies at realistic cost, and
it dies *faster the higher the turnover*. We report gross vs net Sharpe across the ladder,
the break-even cost, and the turnover curve.

Public surface:
- ``data.synthetic_panel`` — the deterministic offline tape: an ``edge`` knob (planted
  predictive strength; ``edge = 0`` is the null where nothing gross survives) and a
  ``persistence`` knob (the AR(1) signal decay that *sets the turnover* — the lever the
  whole demonstration turns on).
- ``strategy`` — the cross-sectional long-short book, the turnover-scaled cost model, the
  cost ladder, the break-even cost, the turnover curve, robust inference, and the
  seed-robust synthetic control.

A pure method demo on a synthetic world by design — cousin of Study 344
(Backtest-Overfitting) and Study 590 (Sharpe-Hacking). It can never earn ``REAL`` (that
needs a robust *t* >= 2 on a real tape); the data-availability limitation is named on the
SIGNAL axis and the study is capped at ``NONE``. The verdict lives on the *third axis* —
a methodology myth-check — not on Signal/Tradability.
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
