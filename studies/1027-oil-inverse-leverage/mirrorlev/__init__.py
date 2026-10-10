"""Study 1027 — Oil Is Equity in a Mirror.

In equities, volatility rises when prices fall — the "leverage effect" Black named in 1976.
The commodity folklore says oil is the mirror image: what moves oil are supply disruptions,
supply disruptions *spike* the price, so in oil it is the **rallies** that bring the storm.
If that were a stable property of the asset, it would have a concrete consequence for anyone
running a volatility-target overlay: on equities the thermostat de-risks after falls, on oil
it would de-risk after rallies.

This study asks whether the mirror is real on the frozen WTI spot tape (1986-2018), whether
it is a permanent property of oil or a regime, and whether it changes what a vol-target
overlay is worth.

- :mod:`mirrorlev.data` — the real tapes (WTI spot daily, S&P 500 price index daily, monthly
  Brent/WTI averages; all frozen inside the ``arch`` package and SHA-256 pinned through
  :mod:`quantlab.bundled`) and the deterministic synthetic GJR generator whose asymmetry sign
  and size are set by ``signal_strength``.
- :mod:`mirrorlev.strategy` — the asymmetry estimators (GJR-GARCH, EGARCH, a model-free
  forward-volatility regression), the regime machinery, skewness, the vol-target overlay and
  the pre-registered verdict.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
