"""Study 397 — Hurst-Regime (does the Hurst exponent time a trend/mean-revert switch?).

The believers' claim: the **Hurst exponent** ``H`` of a price series diagnoses its character
— H > 0.5 is *persistent* (trend-follow), H < 0.5 is *anti-persistent* (mean-revert), H ≈ 0.5
is a random walk — so a rolling-Hurst-gated switch between trend and reversion styles should
beat either style alone and certainly beat running a style in the wrong regime.

We estimate a trailing R/S Hurst on real markets (SPY headline, plus QQQ/GLD/TLT/EFA), build
the trend / revert / Hurst-gated books on a 1-day execution lag net of one-way costs, and ask
— via a block bootstrap of the Sharpe difference and a regime-label placebo — whether
conditioning the style on H adds anything a shuffled label wouldn't.

See :mod:`hurst_regime.data` (real loader + Hurst estimator + deterministic fGn synthetic
control with a planted edge knob) and :mod:`hurst_regime.strategy` (style signals, the gate,
the backtest, and the inference)."""

from . import data, strategy

__all__ = ["data", "strategy"]
