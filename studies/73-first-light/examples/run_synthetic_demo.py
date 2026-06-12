"""Offline, deterministic demo — Study 73 (First-Light).

No network. Builds a synthetic 5-minute tape and shows the study's spine in one screen:
the ORB only beats a random-direction coin when the tape carries a real opening-range
drift — and on a fair open (drift = 0) it is a coin. Run:

    python studies/73-first-light/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from first_light import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 73 — First-Light — ORB synthetic positive control\n")
    print(f"{'planted drift (bps)':>20} | {'ORB bps':>8} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | ORB beats coin?")
    print("-" * 80)
    for drift in (0.0, 10.0, 20.0, 50.0, -20.0):
        bars, _ = data.synthetic_5m(n_days=100, open_drift=drift, seed=73)
        ent = st.orb_entries(bars, range_bars=1)
        if len(ent) == 0:
            print(f"{drift:>20.1f} | {'(no signals)':>8}")
            continue
        orb = st.summarize(
            st.run_trades(bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0),
            "ret_gross",
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, sl_multiplier=1.0, tp_multiplier=None, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if orb["mean_bps"] > rand["mean_bps"] + 0.1 else "no"
        print(f"{drift:>20.1f} | {orb['mean_bps']:>+8.2f} {orb['win_rate']:>6.3f} "
              f"{orb['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} "
              f"| {beats}")

    print("\nThe ORB is a faithful opening-drift harvester: it only edges the coin when")
    print("a real session-start impulse is planted (+10/+20/+50 bps), and on a fair open")
    print("(drift = 0) it is a coin. On the real 60-day tape the signal is not significant.")
    print("See docs/results.md for the live numbers.")


if __name__ == "__main__":
    main()
