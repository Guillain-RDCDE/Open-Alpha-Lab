"""Study 819 — Abnormal-Volume Shock.

Garfinkel & Sokobin (2006): **unusual trading volume** proxies attention / opinion
divergence, and names printing abnormally high volume go on to earn a **positive
subsequent drift**. We sort a liquid US cross-section on its recent **standardised
abnormal volume** (volume vs its own trailing 60-day norm, averaged over a ~5-day
formation window) and measure the forward return of a long-high-abnormal-volume /
short-low-abnormal-volume book.

* ``data``     — the real cross-section (yfinance daily OHLC **+ Volume**, cached under
                 the study's own ``_cache/`` through the ``quantlab.universe`` survivorship
                 guard) plus a deterministic seeded synthetic positive control (a planted
                 attention → drift relation, null at ``edge=0``).
* ``strategy`` — the standardised-abnormal-volume signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
