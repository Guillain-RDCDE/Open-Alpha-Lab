"""Study 841 — Overlapping-Returns Inflation.

A research-method demo (Hansen & Hodrick 1980; Hodrick 1992): predictive regressions that forecast
**overlapping** long-horizon returns (e.g. the next 12 months' cumulative return, sampled monthly)
induce an MA(h-1) autocorrelation in the residuals that grossly inflates the naive OLS t-statistic and
R². On a tape built to have **zero predictability** (``beta = 0``), the naive rejection rate soars far
above the nominal 5% and grows with the horizon, while the Newey-West (HAC) and Hodrick (1992) "1B"
standard errors restore honest inference.

Public surface:
- ``data.simulate_world`` — the deterministic offline monthly world (a Stambaugh-form persistent
  predictor + monthly returns; ``beta = 0`` is the null the demo runs on, ``beta > 0`` plants a
  genuine edge = the positive control).
- ``strategy`` — the overlapping-return builder, the three slope t-stats (naive OLS, Newey-West,
  Hodrick 1B), the naive R², and the Monte-Carlo size/power harness.

A pure method demo on a synthetic world by design — cousin of Study 344 (Backtest-Overfitting) and
Study 590 (Sharpe-Hacking). It can never earn ``REAL`` (that needs a robust *t* ≥ 2 on a real tape);
the data-availability limitation is named on the SIGNAL axis and the study is capped at ``NONE``.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
