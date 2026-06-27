"""Study 522 — Percent Operating Accruals (Hafzalla, Lundholm & Van Winkle 2011).

Percent accruals = operating accruals scaled by **the magnitude of earnings**, not by
total assets (Sloan 1996, Study 231). HLVW (2011) argue this percent-scaling produces a
cross-sectional sort that predicts returns *more strongly* than the Sloan balance-sheet /
asset-scaled accrual, because it ranks firms by the *fraction* of earnings that is
accrual-based rather than the absolute accrual relative to firm size.

    operating accruals      = Net Income - Operating Cash Flow
    percent operating accr. = (Net Income - Operating Cash Flow) / |Net Income|

    HIGH percent accruals => most of earnings is non-cash => LOW future returns
    LOW  percent accruals => earnings are cash-backed       => HIGH future returns

Sort LOW minus HIGH (long cash-backed, short accrual-heavy).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
