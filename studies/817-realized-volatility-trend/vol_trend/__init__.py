"""Study 817 — Realized-Volatility Trend.

The claim under test: sort a cross-section not on the **level** of realized volatility
(the low-vol anomaly, study 330) but on its **trend** — each name's ``(trailing 21d
realized vol) / (trailing 63d realized vol) - 1``. Rising-vol names are said to keep
de-rating, falling-vol names to re-rate, so a long-falling-vol / short-rising-vol book
should earn a positive spread. We ask, honestly, whether that vol-*momentum* is
anything beyond the low-vol *level* effect.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a planted
                 rising-vol->de-rate relation, null at ``edge=0``).
* ``strategy`` — the vol-trend signal, the point-in-time cross-sectional sort, the
                 low-vol-level control + additivity test, the inference primitives
                 (Welch / one-sample / Newey-West HAC / Wilson / placebo), and the
                 costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
