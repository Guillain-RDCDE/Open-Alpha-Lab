"""Study 590 — Sharpe-Hacking.

A research-method demo: how much *reported* Sharpe ratio you can manufacture with pure financial
engineering — **return smoothing** (illiquidity / stale marks), **naive leverage**, and
**volatility targeting** — on a series that carries **zero genuine alpha** by construction. The
parable of metric-gaming: the Sharpe you print is only as honest as the way you measure it.

Public surface:
- ``data.synthetic_series`` — the deterministic offline tape (a single ``alpha`` knob;
  ``alpha = 0`` is a zero-edge series = the null the whole demo runs on, ``alpha > 0`` plants a
  genuine edge = the positive control the engine must respect).
- ``strategy`` — the three Sharpe-gaming transforms, the honest (autocorrelation-corrected) Sharpe,
  a bootstrap/placebo null on the *inflation*, costs, the robustness sweep, and the seed-robust
  synthetic positive control.

A pure method demo on a synthetic world by design — cousin of Study 344 (Backtest-Overfitting) and
Study 589 (Genetic-Algo-Overfit). It can never earn ``REAL`` (that needs a robust *t* ≥ 2 on a real
tape); the data-availability limitation is named on the SIGNAL axis and the study is capped at
``NONE``.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
