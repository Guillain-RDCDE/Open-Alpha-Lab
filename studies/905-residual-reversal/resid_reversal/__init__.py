"""Study 905 — Residual Reversal.

Blitz, Huij, Lansdorp & Verbeek (2013), *"Short-Term Residual Reversal"*: the classic
one-week reversal (last week's losers bounce, winners fade) is badly contaminated by
bid-ask bounce and by common **factor** moves — a name that fell because its whole
sector fell is not "over-sold." Strip the factor first: regress each name's weekly
return on the market, keep the **residual**, and reverse on THAT. The residual reversal
is cleaner, stronger, and — the claim — survives where the raw version dies at the
spread.

* ``data``     — the real cross-section (yfinance daily OHLC + Volume, cached under the
                 study's own ``_cache/`` through the ``quantlab.universe`` survivorship
                 guard) plus a deterministic seeded synthetic positive control (a
                 planted weekly residual mean-reversion, null at ``edge=0``).
* ``strategy`` — the weekly market-model residual, the point-in-time reversal sort with
                 a dollar-volume liquidity screen, the raw-vs-residual comparison, the
                 inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
