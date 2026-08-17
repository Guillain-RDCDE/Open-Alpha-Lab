"""Study 940 — The Turnover Budget.

One cross-sectional momentum sleeve on the eleven Select Sector SPDRs, run at four
rebalance speeds (daily / weekly / monthly / quarterly), priced against the friction it
generates. The deliverable is the **break-even cost per unit of traded notional** for
each speed — the honest turnover budget — not a Sharpe headline.

- :mod:`turnover_budget.data` — the real tape (shared cache, offline loader) and the
  deterministic synthetic panel with a ``signal_strength`` knob.
- :mod:`turnover_budget.strategy` — signal, drifting-weight backtest, costs and borrow,
  the frequency table, the cost and borrow sweeps, and the synthetic control.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
