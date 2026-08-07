"""Study 839 — The t > 3 Threshold.

Harvey, Liu & Zhu (2016), *"... and the Cross-Section of Expected Returns"*: the
published **factor zoo** is the visible tip of an enormous, mostly-unreported search.
When hundreds of candidate factors are data-mined, the conventional **t > 2** bar is far
too lax — a whole paper's worth of "discoveries" is guaranteed by chance alone. The
multiple-testing-adjusted hurdle (Bonferroni / Holm / Benjamini-Hochberg-Yekutieli)
rises toward and beyond **t ~ 3.0**, which is why HLZ recommend a newly claimed factor
clear a *t* of about 3.0, not 2.0.

This is a **research-method** demo on a synthetic world by design:

* ``data``     — a deterministic, seeded **factor zoo** generator. The **null** world is
                 pure noise (every candidate factor is a zero-mean return stream, so any
                 "significant" factor is a false discovery); the **positive control**
                 plants a known subset of genuinely-priced factors so the corrections can
                 be shown to *keep the real ones while dropping the fakes*.
* ``strategy`` — the per-factor *t*-stats (vectorised), the naive t>2 / t>3 threshold
                 counts, the expected false-discovery arithmetic, the family-wise
                 (Bonferroni / Holm) and false-discovery-rate (BH / BHY) cutoffs
                 expressed as an implied ``|t|``, the realized-FDR proof against known
                 truth, the publication haircut, and the seed-robust controls. The house
                 inference primitives (one-sample / Welch / Newey-West / Wilson) travel
                 with it.

A synthetic-only demo can never earn ``REAL`` (that needs a robust *t* >= 2 on a real
tape); the data-availability limitation is named on the SIGNAL axis and the study is
capped at ``NONE``. Cousins on the bench: [346 multiple-testing](../346-multiple-testing/)
(the generic family-wise problem), [536 anomaly-decay-post-publication](../536-anomaly-decay-post-publication/)
(what happens to factors *after* they clear the bar), and
[343 data-mining-roulette](../343-data-mining-roulette/) (mining a single dataset for a
lucky rule).
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
