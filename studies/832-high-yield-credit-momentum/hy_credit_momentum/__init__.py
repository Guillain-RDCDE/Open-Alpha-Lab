"""Study 832 — High-Yield Credit Momentum.

Credit **trend** (trailing total return of HYG in excess of IEF) as a risk-on/off equity
**timing** switch: positive trend → long SPY, negative → de-risk to IEF. Tested net of
costs against a 100%-SPY buy-and-hold, with a Newey-West *t* on the discrimination and on
the active return, a permutation placebo, and a seeded synthetic positive control.
"""
from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
