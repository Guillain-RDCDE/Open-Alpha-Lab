"""Study 589 — Genetic-Algo-Overfit.

A research-method demo: evolve a trading rule with a genetic algorithm until it looks brilliant
in-sample, then watch it die out-of-sample. The parable of overfitting by search — a cousin of
Study 344 (Backtest-Overfitting) and Study 348 (Curve-Fitting), but with the search run by an
*evolutionary optimiser* over a menu of technical features rather than a plain grid.

Public surface:
- ``data.synthetic_series`` — the deterministic offline tape (a single ``signal_strength`` knob;
  0 = pure random walk = null, > 0 = a genuinely predictable tape = positive control).
- ``strategy`` — the GA engine, the IS/OOS split, the Sharpe test, the deflated-Sharpe haircut,
  the label-shuffle placebo, costs, the robustness sweep and the seed-robust control.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
