"""Study 855 — Accrual Quality (Dechow-Dichev).

Dechow & Dichev (2002): earnings whose accruals map **poorly** into past/present/future
operating cash flows are *low-quality* — less persistent, and (the trading claim) they
command a discount. We approximate a name's accrual quality as the **volatility of the
residual** from regressing its (scaled) accruals on operating cash flow over a rolling
window of quarterly filings; a *high* residual vol = *poor* quality. We sort the
cross-section on accrual quality, go **long high-quality (low residual vol) / short
low-quality (high residual vol)**, and ask the three desk questions: does the sort predict
returns, can you trade it, and does poor accrual quality really flag less-persistent
earnings?

* ``data``     — the real tape (EDGAR XBRL ``companyconcept`` fundamentals + yfinance daily
                 closes for a fixed ~45-name basket of deep-history US non-financial filers)
                 plus a deterministic seeded synthetic positive control (a planted
                 quality->return relation, null at ``edge=0``).
* ``strategy`` — the calendar-time tercile long-short (Newey-West HAC), a pooled event-drift
                 cross-check with a label-shuffle placebo, an era split, a costed timer, the
                 earnings-persistence mechanism axis, and the shared inference primitives.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
