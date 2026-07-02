"""Study 568 — Effective-Tax-Rate (the tax-avoidance-as-quality/risk return anomaly).

The accounting claim: a firm's **effective tax rate** (ETR) — income-tax expense over pretax
income — carries information about future returns. The two competing stories:

- **Quality / tax-avoidance premium.** Firms that legally pay a *low* ETR are efficiently
  managed, cash-generative and skilled at shielding income; the market underappreciates the
  durability, so low-ETR names earn *higher* subsequent returns.
- **Risk / red-flag story.** A suspiciously low ETR is a fragile loophole — aggressive tax
  positions that can reverse (IRS/OECD action, one-off shields) — and predicts *lower* returns.

The literature is mixed and the *sign* is the whole question. We build a low-minus-high ETR
sort on real EDGAR fundamentals + yfinance prices, and let the tape decide.

Distinct from this desk's other accounting anomalies: [Study 122 gross-profitability],
[Study 200 ROE-quality], [Study 231 Sloan accruals], [Study 521/522 accruals] scale *earnings*
by assets or cash; Study 568 is the **tax-rate level and its change** as the sort key. Also
distinct from [Study 192 tax-day] (a *calendar* seasonal), which is about April flows, not the
firm-level ETR anomaly.
"""
