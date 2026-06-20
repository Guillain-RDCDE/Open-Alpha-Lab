"""Study 316 — Bank-Failure: is a banking blow-up a buy signal or a warning?

An event study of SPY (and the financials sector) around a hardcoded table of
headline bank-failure dates (Lehman, Bear Stearns, WaMu, SVB, Credit Suisse, …),
with a placebo/synthetic-control arm that re-runs the identical window machinery on
random calendar dates so the harness can prove it detects a planted drift and
otherwise calls noise.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
