"""Offline demo — the reversion machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic mean-reverting panel (and a random-walk null), runs the equal-risk
contrarian book, and shows it recovers the reversion premium GROSS — and that turnover is so high the
edge is fragile the moment costs appear.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rip_tide import costs, data, extension, strategy


def main() -> None:
    r, truth = data.synthetic_reversion(revert_strength=0.06, seed=32)
    r0, _ = data.synthetic_reversion(revert_strength=0.0, seed=32)
    print(f"Synthetic control: {truth.n_markets} markets x {truth.n_days} days, "
          f"revert_strength {truth.revert_strength} (null = 0)\n")
    for nm, rr in [("REVERSION panel", r), ("NULL (random walk)", r0)]:
        g = strategy.summary(strategy.book_returns(rr, cost_bps=0.0))
        n = strategy.summary(strategy.book_returns(rr, cost_bps=2.0))
        print(f"  {nm:20} gross Sharpe {g['sharpe']:5.2f}  |  net@2bp {n['sharpe']:5.2f}  "
              f"turnover/day {strategy.turnover(rr):4.2f}")
    print("\nCost sweep on the reversion panel (the wall):")
    print(costs.cost_sweep(r).round(3).to_string())
    print(f"\nBreak-even cost: {costs.breakeven_cost_bps(r):.2f} bp")
    print("\nHolding-period rescue (slow it down to cut cost):")
    print(extension.holding_period_sweep(r).round(3).to_string())
    print("\nReal-data verdict (18 futures, 2000-2026) is in ../docs/results.md.")


if __name__ == "__main__":
    main()
