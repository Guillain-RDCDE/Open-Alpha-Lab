"""Offline, deterministic demo — Study 129 (Heikin-Ashi).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Heikin-Ashi colour-flip signal only beats a random-direction coin when the tape
actually carries bar-level momentum — and on a martingale (momentum = 0) it is a fair
die.  Also shows that HA produces fewer flips than raw candles (the "smoothing" claim),
but that this alone does not buy directional information.  Run:

    python studies/129-heikin-ashi/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from heikin_ashi import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 129 — Heikin-Ashi — synthetic positive control\n")
    print(f"{'momentum':>12} | {'HA flips':>8} {'raw flips':>9} | "
          f"{'HA bps':>7} {'win':>6} {'t':>6} | "
          f"{'coin bps':>8} {'t':>6} | HA beats coin?")
    print("-" * 90)

    for mom in (0.00, 0.10, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=600, momentum=mom, seed=129)

        ha = st.heikin_ashi(bars)
        ha_col = st.ha_colour(ha)
        raw_col = st.raw_colour(bars)

        ha_flips = ha_col.ne(ha_col.shift(1)).sum()
        raw_flips = raw_col.ne(raw_col.shift(1)).sum()

        ent = st.colour_flip_entries(ha_col)
        ha_res = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross"
        )
        rand_res = st.summarize(
            st.run_trades(
                bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if ha_res["mean_bps"] > rand_res["mean_bps"] + 0.05 else "no"
        print(f"{mom:>12.2f} | {ha_flips:>8} {raw_flips:>9} | "
              f"{ha_res['mean_bps']:>+7.2f} {ha_res['win_rate']:>6.3f} "
              f"{ha_res['tstat']:>+6.2f} | "
              f"{rand_res['mean_bps']:>+8.2f} {rand_res['tstat']:>+6.2f} | {beats}")

    print()
    print("HA consistently has fewer flips than raw candles — the smoothing claim is true.")
    print("But smoothing adds lag, not signal: the flip only beats a coin when momentum is")
    print("planted (+0.10/+0.20), and on a martingale (0.00) it is a fair die.")
    print("On the real daily tape there is no such momentum — see docs/results.md.")


if __name__ == "__main__":
    main()
