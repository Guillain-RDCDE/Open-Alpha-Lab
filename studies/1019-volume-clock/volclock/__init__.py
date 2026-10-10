"""Study 1019 — The Market Runs on Volume Time.

Clark (1973) and Ané & Geman (2000) made one of the most elegant claims in empirical finance:
returns are Gaussian, just not in calendar time. Measured on a clock that ticks with trading
activity, a day of frantic volume is simply *more time* than a quiet one, and the fat tails of
daily returns are what you get when you mix normal draws over a random number of ticks.

This study takes the claim at its word on two daily index tapes, and then asks the question the
elegant version skips: the clock is read at the close, so is any of it usable before?

- :mod:`volclock.data` — the frozen, SHA-pinned S&P 500 and Nasdaq Composite tapes shipped
  inside ``arch`` (via :mod:`quantlab.bundled`), and a deterministic subordinated-process
  generator whose ``signal_strength`` knob decides how much of the variance runs through the
  volume clock.
- :mod:`volclock.strategy` — the measurement (kurtosis removed by each clock, with block-bootstrap
  intervals), the volume–volatility correlation, the out-of-sample forecast race with
  Diebold–Mariano tests, the vol-targeting overlay, and the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
