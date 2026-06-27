"""Study 532 -- Firm-Age-Anomaly: do mature firms beat young, recently-listed firms?

The new-list / firm-age effect (Fama & French 2004; Jiang, Lee & Zhang 2005):
young, recently-IPO'd firms tend to underperform mature, long-established firms. We
proxy firm age from each name's first available price date, sort the basket
old-minus-young each month, and test the premium on a yfinance large-cap panel that
deliberately mixes old-economy survivors with recent IPOs.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
