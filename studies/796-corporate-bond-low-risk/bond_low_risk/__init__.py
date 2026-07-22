"""Study 796 — Corporate-Bond-Low-Risk (Betting-Against-Beta in bonds).

The low-risk anomaly across a small basket of credit and Treasury bond ETFs: each
month-end, rank the universe on its trailing volatility, then build a vol-scaled
low-minus-high spread — long the low-vol names (levered up to a common risk target),
short the high-vol names (levered down) — and ask whether the safe leg delivers more
return per unit of risk. The claim traces to Frazzini & Pedersen (2014), *Betting
Against Beta*: leverage-constrained investors overpay for risk, so low-risk assets earn
the highest risk-adjusted returns.

Two entry points:

* ``data`` — the real ETF tape (yfinance, study-local ``_cache/``) plus a deterministic,
  seeded synthetic panel whose planted low-risk (BAB) knob is null at 0.
* ``strategy`` — the trailing-vol ranking, the vol-scaled long-short book, the costed
  timer, and the inference primitives (HAC one-sample t, Wilson, vol-rank-shuffle placebo).
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
