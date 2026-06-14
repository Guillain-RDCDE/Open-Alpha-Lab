"""Offline, deterministic demo — Study 157 (Kelly-Sizing).

No network. Builds a synthetic daily return series and shows the study's spine
in one screen: Kelly sizing beats fixed fractional only when the underlying tape
carries a real edge — and even then, it comes with much worse drawdowns. Run:

    python studies/157-kelly-sizing/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kelly_sizing import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 157 — Kelly-Sizing — synthetic positive control\n")
    print("Does Kelly beat fixed sizing? Only with a real edge — and at the cost of bigger drawdowns.")
    print()

    for sharpe_plant, label in [(0.0, "zero edge (Sharpe=0)"), (0.5, "SPY-like edge (Sharpe=0.5)")]:
        returns, truth = data.synthetic_returns(n_years=30.0, sharpe=sharpe_plant,
                                                annual_vol=0.16, seed=157)
        f_estimated = st.kelly_fraction_continuous(returns)
        result = st.compare_strategies(returns, train_years=5.0, rebal_freq=21, cost_bps=1.0)

        print(f"--- Planted Sharpe = {sharpe_plant:.1f}  ({label}) ---")
        print(f"  Full-Kelly estimate (in-sample): {f_estimated:.2f}x leverage")
        print()
        print(f"  {'Strategy':<20} {'Ann. Geo. Ret':>14} {'Ann. Vol':>10} "
              f"{'Sharpe':>8} {'Max DD':>10} {'Avg Lev':>9}")
        print("  " + "-" * 76)
        for name, row in result.iterrows():
            print(f"  {name:<20} {row['ann_geo_ret']:>+13.1%} {row['ann_vol']:>9.1%} "
                  f"{row['sharpe']:>8.2f} {row['max_drawdown']:>9.1%} {row['mean_lev']:>9.2f}x")
        print()

    print("Key lessons:")
    print("  1. With a real edge (Sharpe=0.5), full-Kelly maximises growth — but max drawdown")
    print("     is typically 2–3x worse than half-exposure.")
    print("  2. With zero edge (Sharpe=0), Kelly estimates from finite windows are noisy;")
    print("     even small mis-estimates lead to leverage that destroys geometric growth.")
    print("  3. Half-Kelly is the practitioner compromise: ~75% of the growth, ~25% of the variance.")
    print("  See docs/results.md for the real SPY tape verdict.")


if __name__ == "__main__":
    main()
