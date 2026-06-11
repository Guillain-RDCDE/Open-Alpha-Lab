"""Offline demo — the whole study on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic index, runs the vol-targeted contrarian book under both honest
account types (margin: markup on the borrowed slice; futures/CFD: markup on the full notional),
and prints the drawdown protection (real) and the markup bill — the house edge (mirage).
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

from house_edge import costs, data, extension, strategy


def main() -> None:
    price, rate, truth = data.synthetic_market()
    e = strategy.exposure(price)
    print(f"Synthetic control: {truth.n_days} days, {truth.n_crashes} bear regimes, seed {truth.seed}\n")
    rows = {
        "buy & hold (total return)": strategy.summary(costs.buy_and_hold(price), rf=rate),
        "strat — funded flat (0%)": strategy.summary(costs.net_returns(e, price, rate, mode="futures", markup=0.0), rf=rate),
        "strat — margin @ 2.5%": strategy.summary(costs.net_returns(e, price, rate, mode="margin"), rf=rate),
        "strat — CFD @ 2.5%": strategy.summary(costs.net_returns(e, price, rate, mode="futures"), rf=rate),
    }
    print(f"{'':28} {'CAGR':>6} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7}")
    for nm, s in rows.items():
        print(f"{nm:28} {s['cagr']*100:5.1f}% {s['sharpe']:7.2f} {s['max_drawdown']*100:6.1f}% {s['calmar']:7.2f}")
    he = costs.house_edge(e, price, rate)
    dd = extension.drawdown_protection(price, rate)
    print(f"\nReal:   drawdown {dd['bh_max_drawdown']*100:.0f}% -> {dd['strat_max_drawdown']*100:.0f}% "
          f"(a {dd['drawdown_reduction']*100:.0f}-pt reduction)")
    print(f"Mirage: the markup. Funded flat the book makes {he['cagr_funded_flat']*100:.1f}%; the retail CFD "
          f"markup takes {he['house_edge_ann']*100:.2f} pts/yr of it (margin account: {he['margin_edge_ann']*100:.2f}).")


if __name__ == "__main__":
    main()
