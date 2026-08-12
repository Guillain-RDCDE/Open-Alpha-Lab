"""Study 881 — Jobless-Claims Sector Rotation.

The claim: the **4-week change in initial jobless claims** should drive a
**cyclical-vs-defensive sector rotation** — when claims are *rising* (labour market
softening) the market should reward **defensives** (XLP consumer-staples, XLU
utilities) over **cyclicals** (XLY consumer-discretionary, XLI industrials), so the
**cyclical-minus-defensive** return spread should be *depressed*. A labour-nowcast
rotation, not a market-timer (that is study 385's question).

* ``data``     — the real tape: the monthly **4-week-MA initial-claims** level (FRED
                 ``IC4WSA`` / U.S. DoL-ETA, a documented public snapshot, since the FRED
                 CSV host is unreachable in this build) and the four sector-ETF closes
                 plus SPY (yfinance daily, cached under this study's own ``_cache/``),
                 plus a deterministic seeded synthetic control (a planted
                 rising-claims -> cyclicals-underperform relation, null at ``edge = 0``).
* ``strategy`` — the claims-change signal, the cyclical-minus-defensive spread, the
                 predictive Newey-West regression, a permutation placebo, an era cut, a
                 costed rotation timer, and the inference primitives (one-sample /
                 Welch / Newey-West HAC / Wilson / synthetic detector).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
