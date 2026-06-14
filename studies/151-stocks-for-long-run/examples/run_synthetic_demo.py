"""Offline, deterministic demo — Study 151 (Stocks-For-Long-Run).

No network.  Builds a synthetic monthly panel with a planted equity premium
and shows the study's spine in one screen: the claim "stocks always beat bonds
at long horizons" is true in a world with a planted premium, but the worst-case
window still goes negative at short horizons.  Run:

    python studies/151-stocks-for-long-run/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stocks_for_long_run import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 151 — Stocks-For-Long-Run — synthetic positive control\n")
    print(
        f"{'horizon':>8} | {'eq_mean%':>9} {'eq_worst%':>10} {'eq_neg%':>8} "
        f"| {'excess_mean%':>12} {'excess_worst%':>13} {'excess_neg%':>11} "
        f"| {'t_excess':>9}"
    )
    print("-" * 100)

    for premium in (0.05, 0.00):
        label = f"premium={premium:.0%}"
        frame, truth = data.synthetic_world(n_years=155, equity_premium=premium, seed=151)
        print(f"\n  === {label} ===")
        for h in data.HORIZONS:
            r = st.worst_case_windows(frame, h)
            print(
                f"  {h:>6}y | {r['eq_mean']*100:>+9.2f} {r['eq_min']*100:>+10.2f} {r['eq_pct_neg']:>8.1f} "
                f"| {r['excess_mean']*100:>+12.2f} {r['excess_min']*100:>+13.2f} {r['excess_pct_neg']:>11.1f} "
                f"| {r['tstat_excess']:>+9.2f}"
            )

    print("\nWith a planted 5% equity premium:")
    print("  - no 30-year window has a negative real equity return")
    print("  - almost no 30-year window has stocks losing to bonds")
    print("  - the worst-case shrinks dramatically as the horizon lengthens")
    print("\nWith zero premium (null world):")
    print("  - equity still beats bonds ~50% of the time by chance")
    print("  - the 'always beats' claim does NOT hold — the horizon only helps when the premium is real")
    print("\nOn the real Shiller tape (1871-2023): see docs/results.md for the historical record.")


if __name__ == "__main__":
    main()
