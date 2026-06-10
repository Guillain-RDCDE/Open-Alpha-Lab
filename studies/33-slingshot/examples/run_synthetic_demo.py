"""Offline demo — the cross-sectional reversion machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic panel with idiosyncratic overshoot (and a random-walk null), runs the
dollar-neutral reversal book, and shows it recovers the premium GROSS — and that daily turnover makes
the net edge a question of costs.
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

from slingshot import costs, data, extension, strategy


def main() -> None:
    r, _, truth = data.synthetic_panel(revert_strength=0.06, seed=33)
    r0, _, _ = data.synthetic_panel(revert_strength=0.0, seed=33)
    print(f"Synthetic control: {truth.n_stocks} stocks x {truth.n_bars} days, "
          f"revert_strength {truth.revert_strength} (null = 0)\n")
    for nm, rr in [("REVERSION panel", r), ("NULL (random walk)", r0)]:
        g = strategy.summary(strategy.book_returns(rr, cost_bps=0.0))
        n = strategy.summary(strategy.book_returns(rr, cost_bps=5.0))
        print(f"  {nm:20} gross Sharpe {g['sharpe']:5.2f}  |  net@5bp {n['sharpe']:5.2f}  "
              f"turnover/day {strategy.turnover(rr):4.2f}")
    print("\nCost sweep on the reversion panel:")
    print(costs.cost_sweep(r).round(3).to_string())
    print(f"\nBreak-even cost: {costs.breakeven_cost_bps(r):.2f} bp")
    print("\nLookback sweep (the reversal horizon):")
    print(extension.lookback_sweep(r).round(3).to_string())
    print("\nReal-data verdict (S&P 500, 2010-2026) is in ../docs/results.md.")


if __name__ == "__main__":
    main()
