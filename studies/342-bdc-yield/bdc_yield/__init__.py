"""Study 342 — BDC-Yield: do business-development-company yields (BIZD ~10%) survive
credit cycles, or is the headline yield a return of your own capital with a crash
attached?

Business development companies (BDCs) are externally-managed, leverage-funded lenders
to private mid-market firms. The BIZD ETF (VanEck BDC Income) bundles them and quotes a
distribution yield near 10%. We pit BIZD against an equity proxy (SPY) and a bond proxy
(IEF, 7-10y Treasuries) on total return, volatility/drawdown, and — the decisive test —
*who BIZD moves with when credit cracks* (downside beta to stocks vs to bonds, every
>10% equity selloff). Distinct from Study 338 (preferred shares), 340 (bank loans) and
339 (convertibles): BDCs are *levered private-credit equity*, not a senior fixed-income
sleeve — the 10% headline is a distribution rate, not an earned yield.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
