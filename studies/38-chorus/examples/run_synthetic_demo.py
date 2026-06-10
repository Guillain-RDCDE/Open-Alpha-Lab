"""Offline demo — the alpha-combo machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic panel carrying three weak, decorrelated components (and a pure-noise null),
runs each component standalone and the equal-weight & risk-parity combos, and shows the combo Sharpe
beats every soloist while the components are near-uncorrelated — the whole point of the study.
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

from chorus import costs, data, extension, signals, strategy


def main() -> None:
    r, _, truth = data.synthetic_panel(combo_strength=1.0, seed=38)
    r0, _, _ = data.synthetic_panel(combo_strength=0.0, seed=38)
    print(f"Synthetic control: {truth.n_stocks} stocks x {truth.n_bars} days, "
          f"combo_strength {truth.combo_strength} (null = 0)\n")

    sig = signals.all_signals(r)
    print("Standalone components vs the combo (GROSS):")
    print(costs.standalone_vs_combo(sig, r, cost_bps=0.0).round(3).to_string())
    print(f"\nAverage pairwise correlation of components: {strategy.avg_pairwise_corr(sig, r):+.3f}")

    print("\nNull panel — every book should be ~0:")
    sig0 = signals.all_signals(r0)
    print(costs.standalone_vs_combo(sig0, r0, cost_bps=0.0).round(3).to_string())

    print("\nBreadth sweep (combo Sharpe as components are added):")
    print(extension.breadth_sweep(sig, r).round(3).to_string())
    print("\nCost sweep on the equal-weight combo:")
    print(costs.cost_sweep(sig, r).round(3).to_string())
    print("\nReal-data verdict (S&P 500, 2010-2026) is in ../docs/results.md.")


if __name__ == "__main__":
    main()
