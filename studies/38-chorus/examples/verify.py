"""Reproduce the real-data headline run (docs/results.md) — the current S&P 500, 2010–today.

    python examples/verify.py            # cache-only (offline); prints if the panel cache is present
    python examples/verify.py --fetch    # download the S&P 500 panel via quantlab.universe, then run

Builds the three component signals (momentum, reversal, low-vol) on the cached S&P 500 returns panel and
prints each component's standalone Sharpe, the equal-weight and risk-parity combo Sharpe, the average
pairwise correlation of the components, the breadth sweep, the cost sweep, and the repro fingerprint +
as-of that docs/results.md pins.
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


def main(fetch: bool) -> None:
    r = data.fetch_panel(fetch=fetch)
    if r.empty:
        print("No cached panel. Re-run with --fetch (needs network) to download the S&P 500 panel.")
        return
    try:
        from quantlab import repro
        r = repro.as_of(r, "2026-06-10")
    except Exception:
        pass
    r = r.dropna(how="all")

    print(f"\nS&P 500 panel, {r.index[0].date()} - {r.index[-1].date()} ({r.shape[1]} names, {len(r)} days)\n")
    sig = signals.all_signals(r)

    print("Standalone components vs the combo (GROSS):")
    print(costs.standalone_vs_combo(sig, r, cost_bps=0.0).round(3).to_string())

    print(f"\nAverage pairwise correlation of components: {strategy.avg_pairwise_corr(sig, r):+.3f}")
    print("Correlation matrix:")
    print(costs.correlation_matrix(sig, r).round(3).to_string())

    print("\nStandalone vs combo, NET @1bp:")
    print(costs.standalone_vs_combo(sig, r, cost_bps=1.0).round(3).to_string())

    print(f"\nEqual-wt combo turnover/day: {strategy.turnover(strategy.combine(sig, r, scheme='equal')):.2f}"
          f"   break-even: {costs.breakeven_cost_bps(sig, r):.2f} bp")

    print("\nBreadth sweep (combo Sharpe as components are added):")
    print(extension.breadth_sweep(sig, r).round(3).to_string())
    print("\nScheme comparison (equal-wt vs risk-parity, gross):")
    print(extension.scheme_comparison(sig, r).round(3).to_string())
    print("\nCost sweep (equal-wt combo, bp -> Sharpe/CAGR):")
    print(costs.cost_sweep(sig, r).round(3).to_string())

    try:
        from quantlab import repro
        print(f"\nas-of {r.index[-1].date()} · inputs fingerprint {repro.fingerprint(r)}")
    except Exception:
        pass


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
