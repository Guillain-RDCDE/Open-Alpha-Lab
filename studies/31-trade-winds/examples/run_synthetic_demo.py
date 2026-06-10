"""Offline demo — the trend machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic regime-switching trend panel (and a random-walk null), runs the equal-risk
TSMOM book, and shows it recovers the trend premium (and finds nothing in the null).
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

from trade_winds import data, extension, strategy


def main() -> None:
    r, truth = data.synthetic_trends(trend_strength=0.12, seed=31)
    r0, _ = data.synthetic_trends(trend_strength=0.0, seed=31)
    print(f"Synthetic control: {truth.n_markets} markets x {truth.n_days} days, "
          f"trend_strength {truth.trend_strength} (null = 0)\n")
    for nm, rr in [("TREND panel", r), ("NULL (random walk)", r0)]:
        s = strategy.summary(strategy.book_returns(rr, cost_bps=2.0))
        print(f"  {nm:20} Sharpe {s['sharpe']:5.2f}  CAGR {s['cagr']*100:5.1f}%  "
              f"maxDD {s['max_drawdown']*100:5.1f}%  skew {s['skew']:+.2f}")
    print("\nLookback sweep on the trend panel (the premium pays across horizons):")
    print(extension.lookback_sweep(r).round(3).to_string())
    print("\nReal-data verdict (18 futures, 2000-2026) is in ../docs/results.md: "
          "Signal REAL, Tradability FRAGILE standalone, Crisis alpha CONFIRMED.")


if __name__ == "__main__":
    main()
