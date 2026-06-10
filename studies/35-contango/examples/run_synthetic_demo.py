"""Offline demo — the commodity carry / roll-yield machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic term-structure panel (each commodity a persistent roll-yield state that
predicts its return) and a disconnected null, runs the dollar-neutral carry book (long backwardated,
short contangoed), and shows it recovers the premium GROSS — at low turnover, so costs are not the
binding constraint. The real-tape verdict is PENDING a term-structure fetch (see ../docs/results.md).
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

from contango import costs, data, extension, strategy


def main() -> None:
    r, ry, truth = data.synthetic_term_structure(carry_strength=0.9, seed=35)
    r0, ry0, _ = data.synthetic_term_structure(carry_strength=0.0, seed=35)
    print(f"Synthetic control: {truth.n_commodities} commodities x {truth.n_weeks} weeks, "
          f"carry_strength {truth.carry_strength} (null = 0)\n")
    for nm, rr, yy in [("CARRY panel", r, ry), ("NULL (disconnected)", r0, ry0)]:
        g = strategy.summary(strategy.book_returns(rr, yy, cost_bps=0.0))
        n = strategy.summary(strategy.book_returns(rr, yy, cost_bps=5.0))
        pb = strategy.carry_premium_by_bucket(rr, yy)
        print(f"  {nm:22} gross Sharpe {g['sharpe']:5.2f}  net@5bp {n['sharpe']:5.2f}  "
              f"H-L {pb['hml_ann_pct']:+5.1f}%/yr  turnover/wk {strategy.turnover(yy):.3f}")

    print("\nCost sweep on the carry panel:")
    print(costs.cost_sweep(r, ry).round(3).to_string())
    print(f"\nBreak-even cost: {costs.breakeven_cost_bps(r, ry):.1f} bp  (low turnover ⇒ costs not binding)")

    print("\nBeat-7 — carry + momentum blend:")
    c = extension.combine(r, ry, cost_bps=5.0)
    print(f"  carry Sharpe {c['carry_sharpe']:.2f}  momentum Sharpe {c['momentum_sharpe']:.2f}  "
          f"blend Sharpe {c['blend_sharpe']:.2f}  (corr {c['correlation']:+.2f})")

    print("\nReal-tape (front + deferred contracts) is PENDING a term-structure fetch — see ../docs/results.md.")
    basket = data.load_front_month_basket()
    if not basket.empty:
        print(f"(cached front-month basket present: {basket.shape[1]} commodities, "
              f"{basket.index.min().date()}→{basket.index.max().date()} — front-month only, no curve.)")


if __name__ == "__main__":
    main()
