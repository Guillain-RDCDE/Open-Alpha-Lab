"""Offline, deterministic demo — Study 125 (Ichimoku-Cloud).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Ichimoku composite signal (price-vs-cloud + Tenkan/Kijun cross) only beats a
random-direction coin when the tape actually carries bar-level momentum — and on a
martingale (momentum = 0) it is a fair die. Run:

    python studies/125-ichimoku-cloud/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ichimoku_cloud import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 125 — Ichimoku-Cloud — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'signal bps':>10} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | signal beats coin?")
    print("-" * 82)
    for mom in (0.00, 0.10, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=1000, momentum=mom, seed=125)
        ent = st.signal_entries(bars)
        if len(ent) == 0:
            print(f"{mom:>18.2f} | {'(no entries)':>39}")
            continue
        sig = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if sig["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        print(f"{mom:>18.2f} | {sig['mean_bps']:>+10.2f} {sig['win_rate']:>6.3f} "
              f"{sig['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} "
              f"| {beats}  (n={sig['n_trades']})")

    print("\nThe Ichimoku composite signal is a faithful momentum harvester on synthetic data:")
    print("it only edges the coin when momentum is planted (+0.10/+0.20), and on a martingale")
    print("(0.00) it is a fair die. On the real daily tape there is no such persistent")
    print("trend for it to harvest — see docs/results.md.")


if __name__ == "__main__":
    main()
