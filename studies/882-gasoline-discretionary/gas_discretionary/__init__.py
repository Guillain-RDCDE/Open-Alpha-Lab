"""Study 882 — Gas-Price → Discretionary (the "pump tax" rotation).

The claim: a rise in the **gasoline price this month** is a tax on the consumer's wallet, so
it should forecast **consumer-discretionary (XLY) underperforming staples (XLP) next month**
and a **tailwind for energy (XLE)**. We test the self-contained monthly version: a
predictive regression of the discretionary-minus-staples (XLY − XLP) forward one-month
return on the trailing one-month gasoline (RB=F) return, with a Newey-West HAC *t* on the
slope, its sign, and its R².

* ``data``     — the real tape (yfinance daily adjusted close for RB=F + XLY + XLP + XLE +
                 SPY, cached under the study's own ``_cache/``) plus a deterministic seeded
                 synthetic positive control (a planted pump-tax rotation, null at
                 ``edge=0``).
* ``strategy`` — the month-end resample, the discretionary-minus-staples and energy-tilt
                 spreads, the predictive-regression alignment (one documented lag), the
                 inference primitives (OLS+NW HAC / one-sample / Welch / Wilson /
                 permutation placebo), and the costed monthly timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
