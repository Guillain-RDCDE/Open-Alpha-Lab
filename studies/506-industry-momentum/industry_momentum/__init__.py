"""Study 506 -- Industry-Momentum: is cross-sectional momentum mostly an *industry* effect?

Moskowitz & Grinblatt (1999): the bulk of individual-stock momentum is explained by
*industry* momentum -- winner industries keep winning, loser industries keep losing.
We test the sector-ETF embodiment: rank the 11 SPDR sector ETFs each month by their
trailing 12-1 month return, go long the top winners and short the bottom losers, and race
that book against a single-name cross-sectional momentum portfolio built on a large-cap
survivor basket. The question: does sorting *industries* beat sorting *stocks*?
"""

from . import data, strategy

__all__ = ["data", "strategy"]
