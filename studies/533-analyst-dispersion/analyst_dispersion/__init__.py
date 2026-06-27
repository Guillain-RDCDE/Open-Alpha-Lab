"""Study 533 — Analyst-Dispersion (the Diether-Malloy-Scherbina puzzle).

The claim (Diether, Malloy & Scherbina 2002, *Differences of Opinion and the Cross-Section
of Stock Returns*, JF): stocks with **high dispersion in analysts' earnings forecasts** earn
**LOWER** future returns — the opposite of the risk intuition that more disagreement should
demand a higher premium. The mechanism is Miller (1977): when short-sale constraints bind,
prices reflect the *optimists* and high-dispersion names are over-priced, so they
under-perform as the disagreement resolves.

We rebuild the dispersion measure honestly from free data: yfinance's per-name **analyst EPS
estimate spread** for the current fiscal year, ``dispersion = (high − low) / |mean|`` — the
direct cross-sectional analogue of the DMS dispersion measure. We sort a fixed large-cap
basket into dispersion terciles and ask whether high-dispersion names carry *lower* realised
returns.

**The honest data caveat (front and centre).** yfinance exposes only a **current snapshot**
of analyst estimates — there is no historical dispersion panel. So we cannot build a tradable
month-by-month forward sort the way the academic panel does; we can only sort *today's*
dispersion against the *trailing* return realised into the snapshot. That is a contemporaneous
association, not a deployable forward strategy, and it rests on a single cross-section of ~40
names. We name this on the Signal axis and let the verdict fall where it honestly falls.

See :mod:`analyst_dispersion.data` (real snapshot loader + deterministic synthetic control with
a planted dispersion→return knob) and :mod:`analyst_dispersion.strategy` (the dispersion sort,
the high-minus-low long-short, one-sample t, a label-shuffle placebo, costs)."""

from . import data, strategy

__all__ = ["data", "strategy"]
