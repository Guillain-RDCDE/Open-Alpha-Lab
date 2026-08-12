"""Study 899 — Cash + Call "90/10".

Bill **Gross's "90/10"** (and Bodie's "T-bills + calls"): keep ~90% in T-bills so capital is
roughly preserved and spend ~10% on **convex upside** via call options. Because listed-option
history is not freely available, the ~10% sleeve is a **documented proxy** — a rolling **1-year
at-the-money SPY call marked daily with Black–Scholes** (strike = spot at each annual roll, priced
off SPY's trailing realized vol and the ^IRX bill rate; the 10% premium buys as much notional as
that fair price affords, so protection is dearer in high-vol regimes). We test whether the
asymmetric *"protect capital, rent upside"* profile beats buy-and-hold on excess-of-cash Sharpe
across 2008 / 2020 / 2022, net of costs, and whether the call's **convexity** earns anything over a
same-average-exposure static mix.

* ``data``     — the real tape (yfinance daily closes for SPY / BIL total return + ^IRX rate,
                 cached under the study's own ``_cache/``) plus a deterministic seeded synthetic
                 control (a **bear** tape where capital protection visibly holds, a **calm** null
                 where the premium simply bleeds, and an **up-jump** tape where the convex call
                 sleeve out-captures a linear book).
* ``strategy`` — the Black–Scholes-marked 90/10 engine, the buy-and-hold and matched-static
                 benchmarks, the inference primitives (Newey–West HAC / one-sample / Wilson /
                 spanning alpha / block bootstrap), the cost & premium sweeps, the up/down capture
                 asymmetry, and the synthetic detector.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
