"""Study 983 — The Weekend Oracle.

Bitcoin never stops trading. The stock market stops for sixty-five hours every
weekend. So when something happens on a Saturday, crypto prices it and equities cannot — which
makes the weekend crypto move an unusually clean candidate for a genuine leading indicator,
free of the contemporaneous-correlation problem that ruins most lead-lag studies. Unusually
clean is not the same as informative, and this study measures which it is.

- :mod:`weekendoracle.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`weekendoracle.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
