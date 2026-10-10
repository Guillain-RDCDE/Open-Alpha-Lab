"""Study 1022 — Behind the Curve.

The claim, as the macro desks tell it: when the Fed sits *behind the curve* — its policy
rate below what the Taylor (1993) rule prescribes for the inflation and slack it faces —
money is too easy, stocks get a tailwind and long bonds a headwind (inflation risk pushes
yields up). When the Fed is *ahead* of the curve, too tight, sell stocks.

This study builds the Taylor-rule gap the way an investor at the time *could* have built it
— a one-sided (past-only) HP output gap and an Okun-style unemployment gap, every macro print
lagged one quarter for release — and asks whether it forecasts next-quarter equity excess
returns and long-bond yield changes. The gap is extremely persistent, so the inference is the
Stambaugh (1999) small-sample correction plus a simulation null with an AR(1) regressor whose
innovations correlate with returns exactly as in the data. A sign-of-the-gap equity/bills
timing rule, with one lag and costs, is raced against buy-and-hold on excess returns.

The data are **final vintage** (statsmodels ``macrodata``). Orphanides (2001) showed that
real-time output-gap estimates differed badly from the revised ones; every number here is
therefore an *upper bound* on what was knowable, and the study says so wherever it matters.

- :mod:`taylorgap.data` — the bundled, SHA-pinned real tapes (statsmodels ``macrodata``,
  Fama-French monthly total return and T-bills, Moody's AAA yields via ``arch``), the
  duration/convexity long-bond return approximation, and the deterministic synthetic world
  (``signal_strength`` 1 = planted slope, 0 = matched null) the whole test-suite runs on.
- :mod:`taylorgap.strategy` — the Taylor rule, the two real-time gaps, predictive
  regressions with Newey-West, the Stambaugh correction, the simulation null, long-horizon
  overlap-correct inference, the timing backtest, and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
