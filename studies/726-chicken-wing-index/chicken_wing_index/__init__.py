"""Study 726 — Chicken-Wing-Index: the Super-Bowl wing-demand trade on Wingstop (WING).

Public surface:

  * :mod:`chicken_wing_index.data`     — offline synthetic world, cached WING/SPY/^IRX tape,
    and a hardcoded, cited, APPROXIMATE wholesale-wing-price series (a labelled proxy).
  * :mod:`chicken_wing_index.strategy` — per-month HAC t-stats, the Super-Bowl-window Welch
    test, block-bootstrap CI, the 12-month placebo, the January-alpha-vs-SPY regression, and
    the long-January calendar timer vs buy-and-hold.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
