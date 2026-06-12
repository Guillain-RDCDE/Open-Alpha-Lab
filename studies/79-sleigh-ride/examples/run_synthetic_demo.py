"""Offline, deterministic demo — Study 79 (Sleigh-Ride).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Santa window detector only finds an elevated mean when we plant a seasonal drift —
on a martingale (santa_bps = 0) the window is just a random 7-day stretch. Run:

    python studies/79-sleigh-ride/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sleigh_ride import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 79 — Sleigh-Ride — synthetic positive control\n")
    print(
        f"{'planted drift (bps)':>22} | {'window mean':>11} {'win%':>6} {'t(window)':>10} | "
        f"{'diff vs base':>12} {'t(diff)':>8} | signal?"
    )
    print("-" * 90)
    for santa_bps in (0.0, 20.0, 50.0, 80.0, -30.0):
        bars, _ = data.synthetic_daily(n_years=75, santa_bps=santa_bps, seed=79)
        ws = st.santa_window_summary(bars)
        res = st.compare_santa_vs_baseline(bars)
        sig = "yes" if (ws["tstat_vs_zero"] > 2.0 and ws["mean_window_bps"] > 0) else "no"
        print(
            f"{santa_bps:>22.1f} | {ws['mean_window_bps']:>+11.1f} "
            f"{ws['win_rate']:>6.1%} {ws['tstat_vs_zero']:>+10.2f} | "
            f"{res['diff_bps']:>+12.1f} {res['diff_tstat']:>+8.2f} | {sig}"
        )

    print("\nThe detector is a faithful drift finder: it finds the planted Santa")
    print("seasonal only when the drift is real, and reads noise on a null tape.")
    print("On the real ^GSPC tape (1950-2026) see docs/results.md.")


if __name__ == "__main__":
    main()
