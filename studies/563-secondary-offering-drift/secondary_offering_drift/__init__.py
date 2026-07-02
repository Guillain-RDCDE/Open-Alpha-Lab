"""Study 563 — Secondary-Offering-Drift (does a stock keep sliding after it sells fresh shares?).

The corporate-event claim (dilution + negative-signalling): when a company sells a fresh slug of
shares in a **seasoned / secondary equity offering (SEO)**, the stock is supposed to *drift down*
for months afterward — the extra shares dilute existing holders, and management choosing to sell
equity now is a bearish signal (they think the stock is fully valued). This is the mirror image of
the buyback story (Study 368) and a close cousin of the net-share-issuance factor (Study 519):
issuing shares should presage low returns.

We hard-code a transparent table of notable, well-documented **primary/marketed follow-on equity
offerings** (ticker, pricing/announcement date, deal size) and measure the **abnormal**
post-offering drift — the event stock's forward return **minus SPY** over 1/3/6/12 months —
against a matched placebo (the same names on random dates). True point-in-time SEC-424B feeds for
thousands of offerings are not free on yfinance, so the table is an explicit, named sample; the
lesson is statistical (a few-dozen single-name events, dominated by their own variance) and
survives that caveat.

See :mod:`secondary_offering_drift.data` (the event table, real abnormal-return loader, and a
deterministic synthetic positive control with a planted negative-drift knob) and
:mod:`secondary_offering_drift.strategy` (abnormal forward returns with a 1-day execution lag,
Welch t vs a same-names placebo null, win-rate vs the base rate, one-way costs + short borrow on
the short-the-issuer trade)."""

from . import data, strategy

__all__ = ["data", "strategy"]
