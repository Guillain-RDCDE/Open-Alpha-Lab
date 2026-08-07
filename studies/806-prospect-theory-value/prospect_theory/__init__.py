"""Study 806 — Prospect-Theory Value.

Barberis, Mukherjee & Wang (2016): the **cumulative-prospect-theory (TK) value** of a
stock's recent return distribution predicts its future return **negatively**. A
right-skewed, lottery-like tape scores a high TK value; prospect-theory investors
overweight that good gamble and over-pay, so high-TK names go on to earn less. We sort
a liquid US cross-section on its trailing TK value and measure the forward return of a
long-low-TK / short-high-TK book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 negative TK->return relation, null at ``edge=0``).
* ``strategy`` — the Tversky-Kahneman TK value of a return distribution, the
                 point-in-time monthly cross-sectional sort, the inference primitives
                 (Welch / one-sample / Newey-West HAC / Wilson / placebo), and the
                 costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
