"""Offline, deterministic demo — Study 87 (Center-Line).

No network. Builds a synthetic 5-minute tape and shows the study's spine in one screen:
the VWAP-fade only beats a random-direction coin when the tape actually carries intra-
session mean-reversion — and on a martingale (reversion = 0) it is a fair die. Run:

    python studies/87-center-line/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from center_line import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 87 — Center-Line — synthetic positive control\n")
    print(
        f"{'planted reversion':>18} | {'fade bps':>9} {'win':>6} {'t':>6} | "
        f"{'random bps':>10} {'t':>6} | fade beats coin?"
    )
    print("-" * 82)
    for rev in (0.00, 0.15, 0.30, -0.15):
        bars, _ = data.synthetic_5m(n_days=80, reversion=rev, seed=87)
        ent = st.vwap_fade_entries(bars, threshold=0.5)
        if len(ent) == 0:
            print(f"{rev:>18.2f} | (no entries at this threshold)")
            continue
        fade = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if fade["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        print(
            f"{rev:>18.2f} | {fade['mean_bps']:>+9.2f} {fade['win_rate']:>6.3f} "
            f"{fade['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} "
            f"| {beats}"
        )

    print("\nThe VWAP-fade is a faithful mean-reversion harvester: it only edges the coin")
    print("when intra-session reversion is planted, and on a martingale (0.00) it is a")
    print("fair die. On the real 5-minute tape the signal is absent — see docs/results.md.")


if __name__ == "__main__":
    main()
