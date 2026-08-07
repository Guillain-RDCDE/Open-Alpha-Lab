"""Study 824 — the Cochrane-Piazzesi return-forecasting factor.

Cochrane & Piazzesi (2005): a single tent-shaped linear combination of forward rates
forecasts the one-year-ahead excess return of Treasury bonds across all maturities. We
rebuild the factor from the coarse constant-maturity yields yfinance exposes (^IRX,
^FVX, ^TNX, ^TYX) and grade its predictive R^2 / Newey-West t against the average
excess return of the SHY/IEF/TLT bond ETFs.

* ``data``     — the real tape (yfinance daily yields + bond-ETF total-return prices,
                 cached under this study's own ``_cache/``) plus a deterministic seeded
                 synthetic positive control (a planted forward -> excess-return
                 relation, null at ``edge=0``).
* ``strategy`` — forward-rate construction, the average one-year-ahead excess return,
                 the HAC predictive regression / CP factor, the inference primitives
                 (one-sample / Welch / Newey-West HAC / Wilson), an out-of-sample R^2,
                 a block placebo, and a costed duration-timing overlay.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
