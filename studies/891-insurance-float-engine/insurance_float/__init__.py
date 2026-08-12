"""Study 891 — Insurance Float Engine.

P&C insurers earn on **float** — the pool of premiums held between the day a policy is
written and the day claims are paid. Buffett's Berkshire compounded for decades on that
near-zero-cost leverage, so the folklore says a broad insurer basket should be a quiet
structural money-machine: a genuine risk-adjusted edge over the market, *paid for by the
float* rather than by taking financial-sector risk.

We put that to a race. Two liquid insurer wrappers — **KIE** (SPDR S&P Insurance,
equal-weight) and **IAK** (iShares U.S. Insurance) — against **SPY**, both legs measured
**excess-of-cash** (minus **BIL**), with **KBE** (SPDR S&P Bank) as the control that lets us
ask the decisive question: is any insurer out-performance a *float* premium, or just
financial-sector beta wearing a halo?

* ``data``     — the real tape (yfinance daily **total-return** closes for KIE, IAK, SPY,
                 KBE, BIL, cached under this study's own ``_cache/`` as parquet) plus a
                 deterministic seeded synthetic control with a TUNABLE planted edge (null at
                 ``edge_ann = 0``).
* ``strategy`` — monthly returns, the excess-vs-excess Sharpe race, HAC *t* on the return
                 difference, a CAPM and a two-factor (market + bank-spread) decomposition, a
                 bootstrap Sharpe CI, drawdowns, a calendar-year table, an era cut, a
                 one-month-lag rotation probe, and the costed isolation trade.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
