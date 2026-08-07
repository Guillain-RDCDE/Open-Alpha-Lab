"""Study 826 — Treasury Duration BAB.

Frazzini & Pedersen (2014) *betting-against-beta*, run **inside the Treasury
curve** instead of across stocks. We build a duration factor (equal-weight of five
Treasury ETFs, SHY → TLT), estimate each ETF's rolling beta to that factor, then form
the classic BAB book: **long the low-beta (short-duration) legs levered to unit beta,
short the high-beta (long-duration) legs, beta-neutral**. The claim is the low-risk
tilt — the low-beta legs, once levered up, should out-earn the levered-down high-beta
legs.

* ``data``     — the real tape (yfinance daily total-return closes for SHY, IEI, IEF,
                 TLH, TLT, cached under the study's own ``_cache/``) plus a
                 deterministic seeded synthetic positive control (a planted low-beta
                 alpha, null at ``edge=0``).
* ``strategy`` — the duration factor, the rolling betas, the rank-weighted BAB book,
                 the inference primitives (one-sample / Welch / Newey-West HAC / Wilson
                 / permutation placebo), a factor-regression alpha, and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
