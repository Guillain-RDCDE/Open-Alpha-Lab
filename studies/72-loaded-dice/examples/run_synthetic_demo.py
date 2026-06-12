"""Offline, deterministic demo — Study 72 (Loaded-Dice).

No network. Builds a synthetic 5-minute tape and shows the study's spine in one screen:
the SMA(5/10) crossover only beats a random-direction coin when the tape actually carries
bar-level momentum — and on a martingale (momentum = 0) it is a fair die. Run:

    python studies/72-loaded-dice/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loaded_dice import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 72 — Loaded-Dice — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'cross bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | cross beats coin?")
    print("-" * 78)
    for mom in (0.00, 0.10, 0.20, -0.10):
        bars, _ = data.synthetic_5m(n_days=80, momentum=mom, seed=72)
        ent = st.crossover_entries(bars["close"])
        cross = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if cross["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        print(f"{mom:>18.2f} | {cross['mean_bps']:>+9.2f} {cross['win_rate']:>6.3f} "
              f"{cross['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} "
              f"| {beats}")

    print("\nThe cross is a faithful momentum harvester: it only edges the coin when")
    print("momentum is planted (+0.10/+0.20), and on a martingale (0.00) it is a fair die.")
    print("On the real 5-minute tape there is no such momentum — see docs/results.md.")


if __name__ == "__main__":
    main()
