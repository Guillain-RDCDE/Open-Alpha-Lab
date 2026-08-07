"""Study 821 — Turnover Volatility.

Chordia, Subrahmanyam & Anshuman (2001), *"Trading activity and expected stock
returns"*: it is not the **level** of trading activity that carries a premium but its
**variability** — the cross-sectional **coefficient of variation of daily turnover**
predicts returns **negatively**. Names whose turnover is most erratic (a
liquidity-risk cost) command a discount, so a long **low** turnover-vol / short
**high** turnover-vol book should earn a positive spread. We sort a liquid US
cross-section on its trailing CV-of-turnover and measure the forward return of that
book.

* ``data``     — the real cross-section (yfinance daily OHLC **+ Volume**, cached under
                 the study's own ``_cache/`` through the ``quantlab.universe``
                 survivorship guard) plus a deterministic seeded synthetic positive
                 control (a planted negative turnover-vol->return relation, null at
                 ``edge=0``).
* ``strategy`` — the trailing coefficient-of-variation-of-turnover signal, the
                 point-in-time cross-sectional sort, the inference primitives (Welch /
                 one-sample / Newey-West HAC / Wilson / placebo), and the costed timer.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
