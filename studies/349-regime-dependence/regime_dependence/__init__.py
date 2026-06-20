"""Study 349 — Regime-Dependence: a research-method demo on the danger of fitting a
strategy to one era.

This is **not** a teardown of a specific folklore claim — it is a methodology study.
The object under test is a *method of evaluation*: does a single full-sample Sharpe
tell you whether a strategy has a durable edge, or can a strategy that ruled one
decade look like skill while being a regime bet that dies in the next?

The engine builds two strategies side by side on the *same* tape:

- a **durable** edge (a small, stable per-period premium present in every regime), and
- a **regime-fitted** edge (zero in every decade but one, where a large premium was
  planted) — the canonical "ruled the 2010s, was it skill?" trap.

Decade-by-decade (regime-conditional) Sharpe, with Wilson/bootstrap uncertainty on the
*difference* between sub-periods, is the lens that tells them apart. A synthetic
regime-switch control proves the harness can detect the danger; the real tape (a
trend / 60-40 canonical rule) is split by decade to show how it behaves in practice.

Distinct from Study 119 (Real-Rate-Regime, a real-rate timing *rule*) and Study 144
(Permanent-Portfolio, an *allocation* that claims regime robustness): here the regime
split is the *measuring instrument*, and the verdict is a methodology call, not a
trade.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
