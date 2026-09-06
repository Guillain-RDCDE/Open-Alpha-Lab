"""Study 966 — Forecasting Tomorrow's Vol.

Four volatility forecasters compete on the same tape and the same clock: a
21-day rolling standard deviation, RiskMetrics EWMA, a GARCH(1,1) fitted by maximum likelihood,
and Corsi's HAR-RV. Expanding-window refits, strictly out-of-sample, scored with QLIKE and MSE
and compared with a HAC-corrected Diebold-Mariano test.

- :mod:`vol_forecast.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`vol_forecast.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
