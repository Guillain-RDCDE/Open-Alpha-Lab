"""Offline, deterministic demo — Study 131 (Utilities-Canary).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the utilities-canary signal only beats a flat buy-and-hold when the tape actually carries
a planted XLU→SPY predictive relationship — and on the null (canary_signal=0) the
signal is a fair coin that adds nothing.  Run:

    python studies/131-utilities-canary/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities_canary import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 131 - Utilities-Canary - synthetic positive control\n")
    print("Tests whether XLU/SPY relative-strength momentum forecasts forward SPY returns.")
    print("canary_signal < 0 means: XLU RS up => SPY ret down (the folk claim).\n")

    header = (
        f"{'canary_signal':>14} | "
        f"{'Q5-alert 21d':>12} {'Q1-calm 21d':>11} {'spread(bps)':>11} {'t_spread':>9} | "
        f"canary found?"
    )
    print(header)
    print("-" * 75)

    for signal in (0.0, -0.003, -0.005, -0.010, +0.005):
        df, _ = data.synthetic_daily(n_days=6500, canary_signal=signal, seed=131)
        r21 = st.top_minus_bottom(df, horizon=21)
        alert = r21["mean_alert_bps"]
        calm = r21["mean_calm_bps"]
        spread = r21["spread_bps"]
        t = r21["t_spread"]
        found = "yes" if t > 2.0 else "no"
        print(
            f"{signal:>14.3f} | "
            f"{alert:>12.1f} {calm:>11.1f} {spread:>11.1f} {t:>9.2f} | {found}"
        )

    print()
    print("The canary signal is detectable when planted (|signal| >= 0.005, t > 2).")
    print("On the null (signal=0) the spread is noise — a fair baseline.")
    print("On the real tape the machinery is honest: see docs/results.md for the verdict.")


if __name__ == "__main__":
    main()
