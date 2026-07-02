"""Study 570 — Goodwill-Impairment.

The **overpaid-acquisition** claim (accounting): a firm that carries a *bloated goodwill balance*
— goodwill that is large relative to total assets — is a firm that overpaid for its acquisitions,
and that overpayment eventually surfaces as a **goodwill write-down** (an impairment charge). The
market is slow to price the coming write-down, so high-goodwill firms go on to (a) impair at higher
rates and (b) earn **negative** subsequent returns, especially around the write-down announcement.

This study is **synthetic-only**. The real test needs a point-in-time accounting panel
(goodwill / total assets at year *t*, subsequent impairment events, forward returns) that a
no-key retail stack cannot reach — yfinance exposes neither a clean point-in-time goodwill series
nor tagged impairment events. So we build a deterministic, seeded synthetic goodwill panel with a
single knob that plants the effect, prove the engine catches it and stays flat at the null, and
state the data-availability limitation openly on the SIGNAL axis. A synthetic-only study can never
earn `REAL` (that needs a robust *t* ≥ 2 on a REAL tape); it is capped at `WEAK`/`NONE`.

Distinct from this desk's other accounting anomalies: [Study 231 — Sloan accruals](../231-sloan-accruals/)
and [Study 522 — percent operating accruals](../522-percent-operating-accruals/) test the
*accruals* anomaly (earnings quality); Study 570 tests the *goodwill/asset* balance as a predictor
of *impairment events and their return drag* — the overpaid-M&A story, not accrual reversal.
"""
