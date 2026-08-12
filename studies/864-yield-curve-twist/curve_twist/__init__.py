"""Study 864 — Yield-Curve Twist (Butterfly).

Beyond **level** and **slope**, the third mode of the Treasury curve is its
**curvature** — the *butterfly* ``fly = 2*y10 - y5 - y30`` (the belly vs the wings),
and a *twist* is a change in that curvature (``dfly``). This study tests whether the
butterfly level and its change predict forward Treasury (IEF/TLT) and equity (SPY)
returns *distinctly* from the 2s10s slope (studies 66/132) and the roll-down carry
(study 380).

* ``data``     — the real yfinance daily tape (``^FVX``/``^TNX``/``^TYX`` yields plus
                 IEF/TLT/SPY), cache-first under this study's own ``_cache/``, and a
                 deterministic seeded synthetic control with a tunable planted twist
                 edge (null at ``fly_signal = 0``).
* ``strategy`` — the lagged butterfly signal, the HAC predictive regression (with the
                 incremental slope-control dedup test), the quintile-spread sort, the
                 twist (change) variant, a permutation placebo, a costed timing overlay,
                 and the inference primitives.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
