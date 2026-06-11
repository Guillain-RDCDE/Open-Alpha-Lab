"""Study 40 — Paper-Tiger: a published dual-momentum backtest, mirrored back and stress-tested.

The vendor (paperswithbacktest, "Momentum Asset Allocation", SSRN 1585517 = Antonacci's dual
momentum) ships a headline backtest and a Sharpe. We rebuild a faithful **Global Equities Momentum**
(GEM) book on real ETFs, charge real switching costs, and ask the only question that matters: does
the headline survive contact with the right benchmark? Spoiler in the README verdict.
"""

from . import data, strategy  # noqa: F401
