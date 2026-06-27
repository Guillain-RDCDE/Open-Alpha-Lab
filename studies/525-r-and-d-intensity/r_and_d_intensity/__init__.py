"""Study 525 — R-And-D-Intensity 🔬 (the Chan-Lakonishok-Sougiannis R&D premium).

The pitch (and the 2001 *Journal of Finance* headline): firms with high **R&D-to-market-cap**
(R&D / ME) earn higher subsequent returns, because the market under-prices the intangible capital
those expensed R&D dollars build. So a portfolio **long the high-R&D/ME names and short the
low-R&D/ME names** should harvest a persistent *R&D premium*.

We test it the desk's way, in three parts:

* Is there a *statistically real* cross-sectional **R&D/ME premium** on the tape — a long-high /
  short-low spread that clears a HAC ``t >= 2`` **and** survives a label-shuffle placebo — once we
  measure it honestly with a one-year reporting lag and one execution lag?
* If a spread exists, is it tradable — or do monthly turnover **and** short-borrow erase it?
* And the CLS-specific question: does **R&D / market-cap** (the mispricing signal) beat
  **R&D / sales** (the pure spending-intensity signal of study 400), as Chan-Lakonishok-Sougiannis
  claim? Or are both just the growth/tech-vs-value style axis?

See :mod:`r_and_d_intensity.data` (real EDGAR R&D/revenue/shares + yfinance price/return panels,
and a deterministic synthetic positive control with a planted R&D/ME-premium knob) and
:mod:`r_and_d_intensity.strategy` (the two cross-sectional signals, long/short book with one
execution lag, HAC inference, a label-shuffle placebo, costs + borrow)."""

from . import data, strategy

__all__ = ["data", "strategy"]
