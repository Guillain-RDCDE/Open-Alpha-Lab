"""Study 400 — Patent-Intensity 💡 (the "innovation premium").

The pitch: the companies that pour the most into inventing — the highest **R&D-and-patent
intensity** (R&D expense / revenue) — are building tomorrow's moats, so a portfolio that is
**long the most-intensive innovators and short the least-intensive** should harvest an
*innovation premium* the market under-prices. Buy the inventors, short the dinosaurs, collect.

We test it the desk's way, in two halves a label fuses:

* Is there a *statistically real*, cross-sectional **R&D-intensity premium** on the tape — a
  long-high / short-low spread that clears a HAC ``t >= 2`` — once we measure it honestly with a
  one-year reporting lag and no look-ahead in the accounting?
* And if some spread exists, is it a tradable *innovation* factor — or is it **sector and size
  beta in disguise** (R&D-intensive ≈ tech & pharma & semis; R&D-light ≈ banks, staples, energy),
  plus survivorship in a current-membership basket, that costs and shorts erase?

True issued-patent counts are not on a free feed, so we use **reported R&D intensity** (SEC EDGAR
``ResearchAndDevelopmentExpense`` / revenue) as the transparent, audited proxy for "how
patent-and-invention-intensive is this firm" — labelled a proxy throughout, exactly as the
literature (Chan-Lakonishok-Sougiannis 2001; Hirshleifer-Hsu-Li 2013) operationalises it.

See :mod:`patent_intensity.data` (real EDGAR-intensity + yfinance-return panel, and a
deterministic synthetic positive control with a planted innovation-premium knob) and
:mod:`patent_intensity.strategy` (intensity ranking, long/short book with a reporting lag,
HAC inference, costs + borrow, and the sector-neutral control)."""

from . import data, strategy

__all__ = ["data", "strategy"]
