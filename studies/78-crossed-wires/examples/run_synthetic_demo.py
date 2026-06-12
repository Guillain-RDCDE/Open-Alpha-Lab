"""Offline, deterministic demo — Study 78 (Crossed-Wires).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the MACD(12,26,9) signal-line crossover only beats a random-direction coin when the
tape actually carries bar-level momentum — and on a martingale (momentum = 0) it is a
fair die.  This directly mirrors Study 72 (Loaded-Dice, SMA(5/10)) for comparison.

Run:
    python studies/78-crossed-wires/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crossed_wires import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 78 - Crossed-Wires - MACD(12,26,9) synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'n_trades':>8} {'cross bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | cross beats coin?")
    print("-" * 88)
    for mom in (0.00, 0.10, 0.15, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=600, momentum=mom, seed=78)
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
        beats = "yes" if cross["mean_bps"] > rand["mean_bps"] + 0.5 else "no"
        print(
            f"{mom:>18.2f} | {cross['n_trades']:>8d} {cross['mean_bps']:>+9.2f} "
            f"{cross['win_rate']:>6.3f} {cross['tstat']:>+6.2f} | "
            f"{rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} | {beats}"
        )

    print(
        "\nThe MACD cross is a faithful momentum harvester: it edges the coin only when\n"
        "persistence is planted (+0.10/+0.15/+0.20), and on a martingale (0.00) it is a\n"
        "fair die.  On the real daily tape -- see docs/results.md -- signal is WEAK (t=+0.94).\n"
        "\n[Study 72 comparison] SMA(5/10) on 5-minute bars: cross -0.39 bps/trade, HAC t=-1.12.\n"
        "MACD on daily bars uses slower EMAs; low trade count makes the verdict WEAK not NONE."
    )


if __name__ == "__main__":
    main()
