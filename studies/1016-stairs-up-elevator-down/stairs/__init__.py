"""Study 1016 — Stairs Up, Elevator Down.

The trader's proverb says stock indices "take the stairs up and the elevator down": rises are
slow and orderly, falls are fast. Stated carefully, that is not a claim about the *size* of
moves (negative skewness) but about their *order* — a statement that the index path is
**time-irreversible**: run the tape backwards and it would look different, with slow declines
and sudden rallies.

This study measures the claim four ways on three real, frozen tapes (S&P 500 price index
1990-2022, Nasdaq Composite price index 1999-2018, Fama-French market total return 1926-2018):

1. how long it takes to gain X% versus to lose X% inside drawup and drawdown legs;
2. the Ramsey-Rothman time-reversibility statistic with HAC and block-bootstrap errors;
3. skewness by horizon (daily, weekly, monthly) with block-bootstrap intervals, next to a
   quantile skewness that is immune to a handful of crash days;
4. a mechanism check — how much of the asymmetry a GJR / EGARCH model with a leverage term
   reproduces, against a symmetric GARCH and an i.i.d. null that reproduce none of the order.

Then the question an index holder actually has: does knowing the shape pay? A fast-exit /
slow-entry rule, fixed before the run, is raced against its symmetric twin and buy-and-hold,
after costs and with one execution lag.

- :mod:`stairs.data` — the bundled real tapes (``quantlab.bundled``) and the deterministic
  GJR-GARCH synthetic generator whose ``signal_strength`` scales the leverage term.
- :mod:`stairs.strategy` — the legs, the statistics, the nulls, the rules and the verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
