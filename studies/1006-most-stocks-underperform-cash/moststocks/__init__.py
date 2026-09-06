"""Study 1006 — Most Stocks Lose.

Bessembinder (2018) found that the majority of US common stocks underperformed
one-month Treasury bills over their lifetimes, and that the entire net wealth creation of the
US market traced to a few per cent of firms. The result is widely quoted and widely
misunderstood, usually as an argument about stock-picking difficulty. It is really an argument
about **skewness and holding periods**: the median and the mean of a compounding right-skewed
distribution diverge, and they diverge further the longer you compound. This study measures that
divergence directly on a surviving basket — the least favourable sample available — and derives
what it implies for a concentrated portfolio.

- :mod:`moststocks.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`moststocks.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
