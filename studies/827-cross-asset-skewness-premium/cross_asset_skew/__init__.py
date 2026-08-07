"""Study 827 — Cross-Asset Skewness Premium.

The asset-class analogue of the single-name realized-skewness reversal (Study 803):
sort nine asset-class ETFs on their trailing realized return skewness and go long the
low-skew / short the high-skew book. Engine = data (real ETF tape + seeded synthetic
control) + strategy (the monthly cross-sectional sort, HAC inference, placebo, timer).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
