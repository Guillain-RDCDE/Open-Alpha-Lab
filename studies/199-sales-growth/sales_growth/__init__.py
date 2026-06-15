"""Study 199 — Sales-Growth: do high-growth "glamour" stocks underperform?

Lakonishok, Shleifer & Vishny (1994) argue that investors over-extrapolate past
sales growth, bidding up "glamour" (high-growth) stocks and neglecting "value"
(low-growth) stocks. The predicted return: sort on trailing one-year revenue
growth; the low-growth portfolio should outperform the high-growth portfolio.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
