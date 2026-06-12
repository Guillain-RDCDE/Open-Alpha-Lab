"""Offline, deterministic demo — Study 75 (Knee-Jerk).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Connors RSI(2) mean-reversion rule only beats a random-direction coin when the
tape actually carries short-term anti-persistence — and on a martingale (reversion=0)
it is a fair die. Run:

    python studies/75-knee-jerk/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knee_jerk import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 75 — Knee-Jerk — synthetic positive control")
    print("Connors RSI(2) vs a random-direction coin on a daily tape\n")
    print(
        f"{'planted reversion':>18} | {'rule bps':>9} {'t':>6} | "
        f"{'random bps':>10} {'t':>6} | excess | signal?"
    )
    print("-" * 80)
    for rev in (0.00, 0.15, 0.25, -0.10):
        bars, _ = data.synthetic_daily(n_days=2000, reversion=rev, seed=75)
        signal = st.rsi2_entries(bars, rsi_enter=10.0, trend_sma=None)
        rule = st.summarize(
            st.run_trades(bars, signal, cost_bps=0, rsi_exit=60.0, max_hold=10),
            "ret_gross",
        )
        rand = st.summarize(
            st.run_trades(
                bars,
                signal,
                cost_bps=0,
                rsi_exit=60.0,
                max_hold=10,
                random_dirs=st.random_directions(signal.sum(), seed=1),
            ),
            "ret_gross",
        )
        excess = rule["mean_bps"] - rand["mean_bps"]
        # The rule has signal if it clearly outperforms the coin on excess.
        has_signal = "yes" if (excess > 15 and rule["tstat"] > 2.0) else "no"
        print(
            f"{rev:>18.2f} | {rule['mean_bps']:>+9.2f} {rule['tstat']:>+6.2f} | "
            f"{rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} | "
            f"{excess:>+6.1f} | {has_signal}"
        )

    print()
    print("The RSI(2) is a faithful mean-reversion harvester:")
    print(" - excess over random grows with planted anti-persistence.")
    print(" - at reversion=0 (martingale) the excess is small (~+11 bps), noise level.")
    print(" - at reversion=0.25 the excess is ~+43 bps with |t| > 2.")
    print("On the real daily SPY tape see docs/results.md for the pre/post-2009 verdict.")


if __name__ == "__main__":
    main()
