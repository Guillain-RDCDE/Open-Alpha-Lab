"""Study 535 -- Mispricing-Score (Stambaugh-Yu-Yuan 2015).

A COMPOSITE mispricing score: average the cross-sectional rankings of several
well-known anomalies into one combined rank, then sort. The claim (SYY 2015) is
that the composite predicts returns better than any single anomaly, with the
edge concentrated in the short (overpriced) leg.

Public surface:
    data.fetch_panel / data.synthetic_panel / data.fingerprint
    strategy.mispricing_score / strategy.long_short / strategy.summary
    strategy.placebo_null / strategy.synthetic_control
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
