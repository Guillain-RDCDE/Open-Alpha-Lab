"""Study 902 — Multi-Factor Composite.

A live equal-weight sleeve of five iShares single-factor ETFs (VLUE value, QUAL quality,
MTUM momentum, USMV min-vol, SIZE size) rebalanced monthly, raced against SPY on the
excess-of-cash (minus BIL) Sharpe — net of the rebalancing turnover the blend actually
pays. The practitioner's pitch is diversification: single factors take turns working, so a
blend should carry less factor-timing risk. We test whether that diversification also buys
a risk-adjusted *advantage over the market*, honestly, on the real tape.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
