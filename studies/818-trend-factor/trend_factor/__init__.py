"""Study 818 — Trend Factor.

Han, Zhou & Zhu (2016): a stock's expected return is forecast better by a **blend** of
moving-average price signals across many horizons (3, 5, 10, 20, 50, 100, 200 days) than by
any single moving-average timing rule or by momentum alone. We build the normalized MA
signals, estimate rolling cross-sectional slopes (a Fama-MacBeth-lite expectation), dot the
averaged past slopes into today's signals to get the fitted expected return — the **trend
factor** — and sort a liquid US cross-section long-high / short-low trend.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's own
                 ``_cache/`` through the ``quantlab.universe`` survivorship guard) plus a
                 deterministic seeded synthetic positive control (a planted trend->return
                 relation, null at ``edge=0``).
* ``strategy`` — the normalized MA signals, the rolling cross-sectional-slope trend factor,
                 the single-MA and momentum contrasts, the point-in-time sort, the inference
                 primitives (Welch / one-sample / Newey-West HAC / Wilson / placebo), and the
                 costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
