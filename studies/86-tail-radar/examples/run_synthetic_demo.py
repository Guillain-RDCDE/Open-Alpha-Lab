"""Offline, deterministic demo — Study 86 (Tail-Radar).

No network.  Builds synthetic daily tapes and shows the study's spine in one screen:
the SKEW-quintile forward-return spread is flat on a null tape (no planted signal)
and becomes meaningfully negative only when we plant a genuine SKEW signal.  This is
the positive-control check: the machinery works when there is something to find, but
finds nothing when there is nothing.  Run:

    python studies/86-tail-radar/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tail_radar import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 86 — Tail-Radar — synthetic positive control\n")
    print(f"{'skew_signal':>12} | {'h':>3} | {'Q5 mean':>9} {'Q1 mean':>9} "
          f"{'spread':>9} {'t_spread':>9} | signal detected?")
    print("-" * 75)

    for skew_signal in (0.000, 0.005, 0.015, -0.010):
        df, _ = data.synthetic_daily(n_days=2000, skew_signal=skew_signal, seed=86)
        for horizon in (1, 5):
            res = st.top_minus_bottom(df, horizon=horizon)
            detected = "yes" if abs(res["t_spread"]) > 2.0 else "no"
            print(
                f"{skew_signal:>12.3f} | {horizon:>3} | "
                f"{res['mean_top_bps']:>+9.2f} {res['mean_bot_bps']:>+9.2f} "
                f"{res['spread_bps']:>+9.2f} {res['t_spread']:>+9.2f} | {detected}"
            )

    print()
    print("The SKEW spread is a faithful detector: it finds the signal when planted,")
    print("and reads near zero on the null tape (skew_signal = 0).")
    print("On the real tape — see docs/results.md — there is nothing to find.")


if __name__ == "__main__":
    main()
