"""Study 808 — Continuing Overreaction.

Byun, Lim & Yun (2016): a **weighted signed-momentum** score — weight the *signs* of
the more recent monthly returns more heavily — predicts the cross-section of returns
**positively**. A name on a persistent recent up-streak (high "continuing
overreaction", CO) keeps rising; we sort a liquid US cross-section on its CO score and
measure the forward return of a long-high-CO / short-low-CO book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 continuation via a persistent monthly trend state, null at ``edge=0``).
* ``strategy`` — the weighted signed-momentum signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
