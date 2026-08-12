"""Study 896 — Risk-Parity + Trend.

Plain inverse-vol **risk parity** across SPY / TLT / GLD / DBC diversifies but rides
every sleeve through its own bear market. Bolt a **200-day trend gate** onto each sleeve
— hold it only while it is above its 200-day moving average, else its risk budget sits in
cash (BIL T-bills) — and you should de-risk in sustained downtrends. We race RP+trend vs
plain RP **excess-of-cash on both legs** and ask whether the trend gate improves the
Sharpe and the drawdown, net of costs.

* ``data``     — the real six-ETF tape (yfinance daily total-return closes, cached under
                 the study's own ``_cache/``) plus a deterministic seeded synthetic
                 positive control (bull/bear-regime assets; a 200-day gate must not help
                 at ``edge=0`` and must light up at ``edge>0``).
* ``strategy`` — the inverse-vol risk budget, the 200-day trend gate, the monthly
                 one-lag backtest, the excess-vs-excess race, the inference primitives
                 (HAC t / paired Sharpe-diff bootstrap / era cut), the cost sweep, the
                 shuffled-gate placebo and the synthetic control.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
