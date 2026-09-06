"""Study 982 — The Appetite Gauge.

The ratio of high-beta to low-volatility stocks is the market's own risk appetite,
measured without options: when the racy half leads, investors are paying up for risk. It is
quoted daily on trading desks as a leading indicator. The problem is that the ratio's return is
mechanically close to a leveraged bet on the index, so most of what it "predicts" is the
market's own trend — and separating the two is the whole study.

- :mod:`appetite.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`appetite.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
