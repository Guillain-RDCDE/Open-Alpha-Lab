"""Study 892 — Corporate-Bond Ladder.

Engine package: a held-to-maturity bond LADDER (approximated by a duration-staggered
Treasury ETF mix, SHY/IEI/IEF/TLT, held with roll-at-maturity) raced against a
constant-maturity bond FUND (AGG/BND), net of costs, through the 2022 rate shock.

See ``data`` (real-tape + synthetic control) and ``strategy`` (the race, HAC inference,
bootstrap Sharpe CIs, era cuts, calendar-year table and the costed net edge).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
