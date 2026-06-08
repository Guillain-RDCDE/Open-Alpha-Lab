"""Study 06 — Clockwork-Vol: are the VIX's fixed-period cycles a real clock, or shapes
in red noise?

Four modules, one investigation:

    * :mod:`data`       — the gauge (cached ``^VIX``/``^GSPC``) and an offline synthetic
                          series with a **baked-in fixed cycle** the detector must recover.
    * :mod:`spectral`   — the periodogram and the **AR(1) red-noise significance test**
                          (the scientific core: is a peak a cycle, or noise?).
    * :mod:`cycles`     — the theorist's machinery: bandpass, instantaneous phase, turning
                          points, fixed-period projection.
    * :mod:`backtest`   — walk-forward **forecast skill** and the tradeable expression
                          (long the S&P when the VIX cycle is projected to fall).
    * :mod:`robustness` — period drift, red-noise p-values, sub-period skill, bootstrap.

See ``../README.md`` for the seven-beat write-up and ``../../METHODOLOGY.md`` for the desk.
"""

from . import backtest, cycles, data, robustness, spectral  # noqa: F401

__all__ = ["data", "spectral", "cycles", "backtest", "robustness"]
