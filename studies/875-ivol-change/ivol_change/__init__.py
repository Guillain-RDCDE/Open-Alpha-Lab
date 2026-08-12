"""Study 875 — Idiosyncratic-Vol Change.

Distinct from the idiosyncratic-vol **level** puzzle (Ang-Hodrick-Xing-Zhang, study
501): does the **change** in idiosyncratic volatility predict returns? We estimate each
name's residual (market-model) vol over a recent vs a prior window and sort on the delta:
long **falling**-idio-vol, short **rising**-idio-vol, on the theory that a rising
idio-vol (a deteriorating information environment / rising disagreement) precedes lower
returns.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's
                 own ``_cache/`` through the ``quantlab.universe`` survivorship guard)
                 plus a deterministic seeded synthetic positive control (a market factor
                 + a planted rising-idio-vol->lower-return relation, null at ``edge=0``).
* ``strategy`` — the market-model residual-vol signal, its recent-vs-prior change, the
                 point-in-time cross-sectional sort, the inference primitives (Welch /
                 one-sample / Newey-West HAC / Wilson / placebo), the level-vs-change
                 additivity regression, and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
