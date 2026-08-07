"""Study 825 — Oil Predicts Equities.

Driesprong, Jacobsen & Maat (2008), *"Striking Oil: Another Puzzle?"*: a change in the
**oil price this month** predicts **next month's equity return** *negatively* — oil is a
slow-diffusing macro shock, so a rising oil price is a headwind for stocks one month out.
We test the self-contained monthly version: a predictive regression of the S&P 500 (SPY)
forward one-month return on the trailing one-month oil (USO) return, with a Newey-West
HAC *t* on the slope, its sign, and its R².

* ``data``     — the real tape (yfinance daily adjusted close for USO + SPY, cached under
                 the study's own ``_cache/``) plus a deterministic seeded synthetic
                 positive control (a planted negative oil→equity link, null at ``edge=0``).
* ``strategy`` — the month-end resample, the predictive-regression alignment (one
                 documented lag), the inference primitives (OLS+NW HAC / one-sample /
                 Welch / Wilson / permutation placebo), and the costed monthly timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
