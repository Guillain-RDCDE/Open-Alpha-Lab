"""Study 788 — Overnight / Intraday Tug of War.

The Lou-Polk-Skouras (2019) cross-sectional "tug of war": stocks with high past
OVERNIGHT (prev-close -> open) returns keep earning overnight (persistence) while
REVERSING intraday (open -> close), and vice-versa. We sort a liquid US
cross-section on its trailing overnight return and measure the forward overnight
and intraday legs of the sorted portfolios.

Two entry points:

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the
                 study's own ``_cache/`` through the ``quantlab.universe``
                 survivorship guard) plus a deterministic seeded synthetic
                 positive control (a planted tug that is null at ``tug=0``).
* ``strategy`` — the night/day decomposition (via ``quantlab.decompose``),
                 the point-in-time cross-sectional sort, the inference
                 primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
