"""Offline, deterministic demo — Study 132 (Yield-Curve-Steepener).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the yield-curve slope timing rule only beats buy-and-hold when a genuine slope-to-TLT
signal is planted — and on a flat null (slope_signal=0) it adds nothing.  Run:

    python studies/132-yield-curve-steepener/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yield_curve_steepener import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 132 — Yield-Curve-Steepener — synthetic positive control\n")
    print(
        f"{'planted slope_signal':>22} | "
        f"{'Q5-Q1 spread bps':>17} {'t_spread':>9} | "
        f"{'active Sharpe':>13} {'passive Sharpe':>14} | "
        f"spread > 0?"
    )
    print("-" * 90)

    for slope_sig in (0.000, 0.003, 0.007, 0.012):
        df, _ = data.synthetic_daily(n_days=5000, slope_signal=slope_sig, seed=132)
        tmb = st.top_minus_bottom(df, horizon=1)
        ov = st.timing_overlay(df, cost_bps=0.0)
        active_s = st.summarize(ov["r_active"])
        passive_s = st.summarize(ov["r_passive"])
        beats = "yes" if tmb["spread_bps"] > 0.5 else "no"
        print(
            f"{slope_sig:>22.3f} | "
            f"{tmb['spread_bps']:>+17.2f} {tmb['t_spread']:>+9.2f} | "
            f"{active_s['sharpe_ann']:>+13.3f} {passive_s['sharpe_ann']:>+14.3f} | "
            f"{beats}"
        )

    print()
    print("The engine works: when a genuine slope signal is planted, the Q5-Q1 spread")
    print("grows and the active arm overtakes passive. On the null (0.000), nothing is")
    print("found. On the real tape, t_spread < 1 — see docs/results.md.")


if __name__ == "__main__":
    main()
