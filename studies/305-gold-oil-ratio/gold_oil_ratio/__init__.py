"""Study 305 — Gold-Oil-Ratio: the gold/oil ratio as a risk-on/risk-off equity-timing switch.

A distinct angle from its two cousins on the desk:
- Study 85 (Dr-Copper) runs *predictive regressions* of the copper/gold ratio change on
  forward equity returns (does the ratio forecast SPY?).
- Study 113 (Gold-Silver-Ratio) trades a *z-score pairs / mean-reversion* between two metals.

This study asks a third, different question: used as a binary **market-timing switch** — hold
SPY when the gold/oil ratio says "risk-on", sit in cash (the risk-free rate) when it says
"risk-off" — does the gold/oil ratio beat a passive buy-and-hold of SPY on an
**excess-of-cash, risk-adjusted** basis? That is the way "Dr. Copper times equities" folklore
is actually pitched, applied to gold/oil.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
