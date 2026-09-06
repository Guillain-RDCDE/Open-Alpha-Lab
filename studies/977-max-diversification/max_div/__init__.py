"""Study 977 — Maximum Diversification.

Choueifaty and Coignard's "most diversified portfolio" maximises the ratio of the
weighted-average volatility of the holdings to the volatility of the portfolio itself. It is a
different objective from minimum variance, it has a tidy theoretical story, and TOBAM built a
business on it. This study asks the two questions the story does not answer: is the resulting
portfolio actually different, and does it do better out of sample than the alternatives that
need no optimiser at all?

- :mod:`max_div.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`max_div.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
