"""Offline, deterministic demo — Study 111 (VIX-Term-Structure).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the VIX term-structure slope signal (log(VIX3M/VIX)) only produces a meaningful Q5-Q1
forward-return spread when predictive power is planted in the tape — and on the null
(contango_signal=0) the spread is indistinguishable from zero.  Run:

    python studies/111-vix-term-structure/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vix_term_structure import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 111 — VIX-Term-Structure — synthetic positive control\n")
    print(f"{'signal strength':>16} | {'Q5 bps':>8} {'Q1 bps':>8} {'spread bps':>11} "
          f"{'t-spread':>10} | finds edge?")
    print("-" * 80)

    for sig in (0.00, 0.02, 0.05, 0.10):
        df, _ = data.synthetic_daily(n_days=3000, contango_signal=sig, seed=111)
        res = st.top_minus_bottom(df, horizon=1)
        found = "yes" if res["spread_bps"] > 2.0 and res["t_spread"] > 1.5 else "no"
        print(
            f"{sig:>16.2f} | {res['mean_top_bps']:>+8.2f} {res['mean_bot_bps']:>+8.2f} "
            f"{res['spread_bps']:>+11.2f} {res['t_spread']:>+10.2f} | {found}"
        )

    print()
    print("Timing overlay — contango=long, backwardation=flat — vs buy-and-hold:")
    print(f"{'signal':>16} | {'active bps/d':>13} {'passive bps/d':>14} {'spread t':>10}")
    print("-" * 60)
    for sig in (0.00, 0.05):
        df, _ = data.synthetic_daily(n_days=3000, contango_signal=sig, seed=111)
        ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)
        sa = st.summarize(ov["r_active"])
        sp = st.summarize(ov["r_passive"])
        ss = st.summarize(ov["r_spread"])
        print(
            f"{sig:>16.2f} | {sa['mean_bps']:>+13.2f} {sp['mean_bps']:>+14.2f} {ss['t_stat']:>+10.2f}"
        )

    print()
    print("The slope signal is a faithful edge-finder: with planted signal the Q5-Q1 spread")
    print("turns on; on the null (0.00) the spread is noise around zero.")
    print("On the real tape see docs/results.md — which is what actually matters.")


if __name__ == "__main__":
    main()
