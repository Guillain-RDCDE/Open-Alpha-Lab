"""Study 870 — Industry-Leader Lead-Lag.

Kewei Hou (2007): information diffuses **within an industry** from the *biggest* name
outward. The largest-cap firm in a sector prices news first; the smaller **followers**
react with a lag, so the leader's return this week predicts the followers' return next
week. We long the followers whose leader rose and short those whose leader fell, on a
liquid US cross-section, and cost the book.

* ``data``     — the real cross-section (yfinance daily OHLCV, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard),
                 a static GICS-style sector map + largest-cap ``LEADERS`` designation,
                 plus a deterministic seeded synthetic positive control (a planted
                 leader→follower weekly diffusion, null at ``edge=0``).
* ``strategy`` — the weekly-return panel, the point-in-time leader→follower sort, the
                 inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
