"""Study 888 — CLO AAA Carry.

Does a **AAA-rated CLO tranche** (JAAA since 2020-10, ICLO since 2022-12) pay a *real,
mechanical, risk-adjusted* pickup over cash and over same-rated IG corporates (LQD) and
duration (IEF) — a structural-complexity / seniority premium — and does it survive costs?

* ``data``     — the real tape (yfinance total-return closes for JAAA/ICLO/LQD/IEF/BKLN/BIL,
                 cached under the study's own ``_cache/``) plus a deterministic seeded
                 synthetic positive control (a planted excess-of-cash carry, null at
                 ``carry_annual=0``, with a high-vol duration decoy).
* ``strategy`` — the excess-of-cash carry stats (Sharpe + block-bootstrap CI + HAC *t* +
                 drawdown), the excess-vs-excess race across legs, the JAAA-vs-benchmark
                 head-to-heads, the ZIRP-vs-high-rate era cut, and the costed harvest /
                 relative-trade tradability. HAC and bootstrap reuse ``quantlab``.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
