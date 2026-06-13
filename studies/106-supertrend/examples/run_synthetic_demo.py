"""Offline, deterministic demo — Study 106 (Supertrend).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Supertrend flip only beats a random-direction coin when the tape actually carries
bar-level momentum — and on a martingale (momentum = 0) it is a fair die.  Run:

    python studies/106-supertrend/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supertrend import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 106 — Supertrend — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'flip bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | flip beats coin?")
    print("-" * 82)
    for mom in (0.00, 0.10, 0.30, 0.50, -0.10):
        bars, _ = data.synthetic_daily(n_days=1000, momentum=mom, seed=106)
        ent = st.flip_entries(bars, atr_n=10, mult=3.0)
        flip = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if flip["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        print(f"{mom:>18.2f} | {flip['mean_bps']:>+9.2f} {flip['win_rate']:>6.3f} "
              f"{flip['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} "
              f"| {beats}")

    print("\nThe Supertrend flip is a faithful momentum harvester: it only edges the")
    print("coin when strong momentum is planted (>=0.30), and on a martingale")
    print("(0.00) it is a fair die.  On the real daily tape there is at best weak")
    print("and intermittent trend -- see docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
