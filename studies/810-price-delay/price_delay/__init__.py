"""Study 810 — Price Delay.

Hou & Moskowitz (2005): stocks into which market information diffuses **slowly** — whose
return responds to the market with a lag — earn a **return premium** over stocks that
price the same information promptly. We build the per-name **delay** measure from a
weekly regression of a stock's return on the contemporaneous market plus 4 weekly lags
(``delay = 1 - R2_restricted / R2_unrestricted``) on a liquid US cross-section, then
measure the forward return of a long-high-delay / short-low-delay book.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 slow-diffusion premium, null at ``knob=0``).
* ``strategy`` — the weekly delay regression, the point-in-time cross-sectional sort, the
                 inference primitives (Welch / one-sample / Newey-West HAC / Wilson /
                 placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
