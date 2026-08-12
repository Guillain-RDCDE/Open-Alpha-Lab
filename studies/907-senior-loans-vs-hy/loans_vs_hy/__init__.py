"""Study 907 — Senior Loans vs High-Yield: is the "seniority premium" a real edge?

The pitch, steelmanned: *"Senior secured bank loans (BKLN, SRLN) sit **above** high-yield
bonds (HYG, JNK) in the capital stack at a similar yield — first lien, better recovery,
floating coupon. So a loan sleeve pays you the same carry as junk bonds with less risk: a
**seniority premium** collected for free."* We race the loan sleeve against the HY sleeve,
both **excess of cash** (BIL), on the axes that decide it:

- **The risk-adjusted race (the claim)** — does the loan leg earn a genuinely higher
  excess-Sharpe, robust to a bootstrap and stable across sub-eras?
- **The return premium (the carry claim)** — is loans-minus-HY positive and HAC-significant,
  or do loans quietly earn *less*?
- **Where seniority helps and where it bites** — floating rate + first lien cushion
  rate/spread selloffs (2015-16 energy, 2022), but the loan sleeve is the *less liquid* leg
  and can gap **worse** in a pure liquidity crisis (2020).
- **Tradability** — long-loans / short-HY, charged one-way cost × NAV per rebalance + borrow.

Distinct from Study 340 (Bank-Loans — loans vs *rates*, the duration story) and Study 115
(Credit-Spreads): this is a **loans-vs-HY seniority race** inside the sub-investment-grade
credit box. As-of 2026-06-30.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
