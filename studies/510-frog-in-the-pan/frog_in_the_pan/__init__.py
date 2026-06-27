"""Study 510 -- Frog-In-The-Pan (FIP): is momentum stronger when the past return arrived gradually?

Da, Gurun & Warachka (2014): the market under-reacts more to information that arrives in many
small continuous steps than to the same total move delivered in a few jumps. They proxy this
"information discreteness" (ID) by the sign-consistency of daily returns and show that momentum
is concentrated in the LOW-ID (gradual) names: a frog boiled slowly never jumps out of the pan.

We test the interaction on a survivorship-biased large-cap basket using yfinance daily prices:
double-sort momentum x ID, race the low-ID WML book against the high-ID WML book, charge real
costs and short borrow, run a label-shuffle placebo, and prove engine faithfulness on a planted
synthetic positive control.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
