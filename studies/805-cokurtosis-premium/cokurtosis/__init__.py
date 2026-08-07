"""Study 805 — Cokurtosis Premium.

Fang & Lai (1997) / four-moment CAPM: a security's **systematic kurtosis** — its
**cokurtosis** with the market, ``E[(r_i-μ_i)(r_m-μ_m)^3]/(σ_i·σ_m^3)`` — is a priced
risk. High cokurtosis (a name that amplifies the market's fat-tailed moves) should
command a **positive** premium, so a long high-cokurtosis / short low-cokurtosis book
should earn a positive spread.

* ``data``     — the real cross-section (yfinance daily OHLC, cached under the study's own
                 ``_cache/`` through the ``quantlab.universe`` survivorship guard) plus a
                 deterministic seeded synthetic positive control (a planted positive
                 cokurtosis->return relation, null at ``knob=0``).
* ``strategy`` — the trailing cokurtosis-with-the-market signal, the point-in-time
                 cross-sectional sort, the inference primitives (Welch / one-sample /
                 Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
