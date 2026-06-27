"""Study 526 — Intangible-Value 🧠 (the Lev-Srivastava intangibles-adjusted value premium).

The pitch (Lev & Srivastava 2019; Eisfeldt & Papanikolaou 2013; Peters & Taylor 2017): plain
**book-to-market** (B/M) value investing has decayed because GAAP *expenses* the intangible
capital — R&D, brand, organisational know-how — that increasingly drives firm value. Reported book
equity therefore *understates* the true capital of intangible-heavy firms, contaminating the B/M
value sort. The fix: **capitalise** historical R&D and a share of SG&A into an *intangible-adjusted
book* and re-run the value sort. The claim is that the adjusted-B/M long-short beats plain B/M.

We test it the desk's way, in three parts:

* Is there a *statistically real* cross-sectional **intangible-adjusted-B/M premium** on the tape —
  a long-cheap / short-expensive spread that clears a HAC ``t >= 2`` **and** survives a
  label-shuffle placebo — once we measure it honestly with a one-year reporting lag and one
  execution lag?
* If a spread exists, is it tradable — or do monthly turnover **and** short-borrow erase it?
* And the Lev-Srivastava-specific question (the third axis): does **intangible-adjusted B/M** beat
  **plain B/M** on the *same* basket, as the intangibles-correction thesis claims? Or is the
  adjustment cosmetic on a modern large-cap tape?

See :mod:`intangible_value.data` (real EDGAR book-equity/R&D/SG&A/shares + yfinance price/return
panels, and a deterministic synthetic positive control with a planted value-premium knob) and
:mod:`intangible_value.strategy` (the two cross-sectional B/M signals, long/short book with one
execution lag, HAC inference, a label-shuffle placebo, costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
