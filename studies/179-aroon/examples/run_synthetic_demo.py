"""Offline, deterministic demo -- Study 179 (Aroon).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Aroon(25) crossover edges a random-direction coin only when the tape carries
bar-level trend persistence -- and on a martingale (trend = 0) it is a fair die.  Run:

    python studies/179-aroon/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aroon import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 179 -- Aroon -- synthetic positive control\n")
    print(
        f"{'planted trend':>14} | {'cross bps':>9} {'win%':>6} {'t':>6} | "
        f"{'random bps':>10} {'t':>6} | edge (cross-coin)"
    )
    print("-" * 80)

    for trend_val in (0.00, 0.25, 0.50, -0.25):
        bars, _ = data.synthetic_daily(n_days=2000, trend=trend_val, seed=179)
        ar = st.aroon(bars, period=25)
        ent = st.crossover_entries(ar)
        if len(ent) < 5:
            print(f"{trend_val:>14.2f} | (too few entries)")
            continue
        cross = st.summarize(
            st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, hold_days=5, cost_bps=0,
                directions=st.random_directions(len(ent), seed=7),
            ),
            "ret_gross",
        )
        edge = cross["mean_bps"] - rand["mean_bps"]
        print(
            f"{trend_val:>14.2f} | {cross['mean_bps']:>+9.2f} {cross['win_rate']*100:>5.1f}%"
            f" {cross['tstat']:>+6.2f} | {rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f}"
            f" | {edge:>+.2f} bps"
        )

    print()
    print("The crossover advantage over the coin increases with planted trend persistence.")
    print("On the real daily tape the signal is WEAK -- pooled t=+3.25 but cross-sectionally")
    print("inconsistent (SPY t=0.74, QQQ t=1.48, IWM t=2.95).  See docs/results.md.")


if __name__ == "__main__":
    main()
