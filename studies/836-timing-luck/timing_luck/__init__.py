"""Study 836 — Rebalance Timing Luck.

A research-method demo (Hoffstein, Sober & Vezeris 2019): the **same** monthly
cross-sectional momentum book, rebalanced on a **different day of the period**, produces
materially different equity curves and Sharpe ratios — a *phantom dispersion* that is
pure luck, not skill. The fix is **tranching / overlapping portfolios**, which collapses
the dispersion to a single curve.

Public surface:
- ``data.synthetic_panel`` — the deterministic offline return panel (a single
  ``mom_edge`` knob; ``mom_edge = 0`` is a zero-momentum-edge world = the null the demo
  runs on, ``mom_edge > 0`` plants a genuine momentum premium = the positive control).
- ``strategy`` — the momentum signal, the per-offset book, the timing-luck dispersion,
  the tranched/overlapping fix, the out-of-sample offset-persistence check, the
  inference primitives, a costed timer, and the seed-robust synthetic control.

A pure method demo on a synthetic world by design — cousin of Study 344
(Backtest-Overfitting) and Study 590 (Sharpe-Hacking). It can never earn ``REAL`` (that
needs a robust *t* ≥ 2 on a real tape); the data-availability limitation is named on the
SIGNAL axis and the study is capped at ``NONE``.
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
