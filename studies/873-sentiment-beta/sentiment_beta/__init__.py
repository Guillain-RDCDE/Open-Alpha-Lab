"""Study 873 — Sentiment Beta.

Baker & Wurgler (2006, 2007): the stocks whose returns **co-move most with market
sentiment** — high *sentiment beta* — are the speculative, hard-to-value names that get
over-priced in euphoria and **under-perform afterwards**. We sort a liquid US
cross-section on each name's beta to a tradable sentiment gauge and measure the forward
return of a long-low-beta / short-high-beta book, and its dependence on the sentiment
level.

* ``data``     — the real cross-section (yfinance daily OHLCV, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a common
                 sentiment factor with dispersed per-name loadings, and a planted
                 high-beta->low-return relation, null at ``edge=0``).
* ``strategy`` — the tradable high-minus-low-vol sentiment gauge, the rolling
                 sentiment-beta signal, the point-in-time cross-sectional sort, the
                 post-peak conditional, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
