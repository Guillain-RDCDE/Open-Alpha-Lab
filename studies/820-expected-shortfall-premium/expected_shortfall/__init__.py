"""Study 820 — Expected-Shortfall Premium.

A downside tail-risk premium: stocks whose recent daily returns have a fat **left
tail** — a large trailing **Expected Shortfall** (CVaR) at 5%, the mean of the worst
5% of days — are riskier, so if tail risk is priced they should earn **more**. We sort
a liquid US cross-section on its trailing-year historical ES and measure the forward
return of a long-high-ES / short-low-ES book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 priced downside-tail-risk premium, null at ``edge=0``).
* ``strategy`` — the trailing Expected-Shortfall signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
