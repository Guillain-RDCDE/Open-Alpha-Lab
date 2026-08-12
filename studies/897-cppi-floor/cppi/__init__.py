"""Study 897 — CPPI Floor.

Black & Jones (1987) / Perold & Sharpe (1988): **Constant-Proportion Portfolio Insurance**.
Hold a fixed **multiplier** ``m`` times the **cushion** (portfolio value minus a protected
**floor** that accretes at the cash rate) in equities (SPY), the rest in bills (BIL),
rebalancing daily as the cushion moves — a mechanical, self-financing downside floor. We test
whether the floor actually holds across 2008 / 2020 / 2022, and what the insurance costs in
upside (excess-of-cash Sharpe) versus buy-and-hold and versus a same-average-exposure static
mix, net of costs and gap risk.

* ``data``     — the real tape (yfinance daily total-return closes for SPY / IEF / BIL, cached
                 under the study's own ``_cache/``) plus a deterministic seeded synthetic
                 control (a **bear** tape where the mechanical floor visibly holds, a **calm**
                 null where nothing draws down, and a crash-gap knob for the floor-breach test).
* ``strategy`` — the CPPI engine, the buy-and-hold and matched-static benchmarks, the
                 inference primitives (Newey–West HAC / one-sample / Welch-style / Wilson /
                 spanning alpha / block bootstrap), the cost & multiplier sweeps, the
                 gap-risk stress, and the synthetic detector.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
