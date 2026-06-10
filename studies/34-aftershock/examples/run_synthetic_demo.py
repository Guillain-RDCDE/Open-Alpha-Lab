"""Offline demo — the PEAD machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic stock panel where earnings surprises predict the next weeks of drift (and a
null where the same surprises are noise), runs the dollar-neutral long-positive/short-negative-surprise
book, and shows it recovers the drift GROSS on the control and ~nothing on the null — plus the
surprise-signed drift-decay curve (the PEAD shape) and the holding-period sweep.

The real PEAD run needs a long history of earnings dates + surprises, which no free source supplies in
this sandbox — see examples/verify.py and ../docs/results.md (the run is pre-registered and pending).
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

from aftershock import costs, data, extension, strategy


def main() -> None:
    panel, events, truth = data.synthetic_pead(seed=34)
    p0, e0, _ = data.synthetic_pead(drift_strength=0.0, seed=34)
    print(f"Synthetic control: {truth.n_stocks} stocks x {truth.n_bars} days, {len(events)} earnings events, "
          f"drift_strength {truth.drift_strength} (null = 0)\n")
    for nm, (pp, ee) in [("DRIFT panel (control)", (panel, events)), ("NULL (surprise = noise)", (p0, e0))]:
        g = strategy.summary(strategy.book_returns(pp, ee, cost_bps=0.0))
        n = strategy.summary(strategy.book_returns(pp, ee, cost_bps=5.0))
        print(f"  {nm:24} gross Sharpe {g['sharpe']:5.2f}  |  net@5bp {n['sharpe']:5.2f}  "
              f"turnover/day {strategy.turnover(pp, ee):4.3f}")
    print("\nCost sweep on the drift panel:")
    print(costs.cost_sweep(panel, events).round(3).to_string())
    print(f"\nBreak-even cost: {costs.breakeven_cost_bps(panel, events):.1f} bp")
    print("\nHolding-period sweep (does the drift persist long enough to pay?):")
    print(extension.holding_period_sweep(panel, events).round(3).to_string())
    curve = extension.drift_decay_curve(panel, events, window=70)
    print("\nDrift-decay curve (mean surprise-signed cumulative return), days since announcement:")
    for d in (0, 10, 20, 40, 60, 69):
        print(f"  day {d:2d}: {curve.iloc[d]:+.4f}")
    print("\nReal-data verdict (PEAD on the tape) is PENDING an earnings-history fetch — see ../docs/results.md.")


if __name__ == "__main__":
    main()
