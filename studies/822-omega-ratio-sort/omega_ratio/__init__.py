"""Study 822 — Omega-Ratio Sort.

Keating & Shadwick (2002): the **Omega ratio** at threshold 0 —
``Ω(0) = E[max(r,0)] / E[max(−r,0)]`` — is a gain/loss ratio that summarises the
*whole* return distribution (every moment), not just mean and variance. The pitch:
sorting a cross-section on trailing Omega (long high-Omega / short low-Omega) should
beat a plain trailing-Sharpe sort because it "sees" skewness and fat tails that Sharpe
throws away. We put that head-to-head against Sharpe on a liquid US cross-section.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 Omega->return relation via a vol tilt, null at ``edge=0``).
* ``strategy`` — the trailing Omega(0) signal, the Sharpe and low-vol comparators, the
                 point-in-time cross-sectional sort, the inference primitives (Welch /
                 one-sample / Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
