"""Study 866 — Flight-to-Quality Beta.

Which stocks are the *true* defensives — the names that actually **rally when Treasuries
rally on risk-off days**? For each name we estimate its beta to the **TLT** (long
Treasury) daily return *conditional on down-SPY days* — a **flight-to-quality beta**
(``beta_ftq``). The CAPM-of-insurance prediction: a name with a high FTQ beta is a good
crash hedge, so investors overpay for it and it should earn a **lower** average return
(you pay an insurance premium), while delivering **real crash protection** (a smaller
drawdown on the worst market days).

* ``data``     — the real cross-section (yfinance daily OHLC, 50 liquid US mega-caps,
                 cached under this study's own ``_cache/`` through the
                 ``quantlab.universe`` survivorship guard) plus the **TLT** long-Treasury
                 and **SPY** market proxy closes, and a deterministic seeded synthetic
                 positive control (a planted FTQ-beta -> low-return relation, null at
                 ``edge=0``).
* ``strategy`` — the conditional flight-to-quality beta, the point-in-time monthly
                 cross-sectional sort (long low-FTQ / short high-FTQ, betting-against-
                 insurance), a crash-day drawdown comparison, and the inference
                 primitives (Welch / one-sample / Newey-West HAC / Wilson / placebo /
                 costed timer / synthetic detector).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
