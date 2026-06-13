"""Offline, deterministic demo — Study 116 (Power-Hour).

No network. Builds synthetic session panels with a known last-hour continuation knob and
shows the study's spine in one screen: the follow-morning signal beats a random-direction
coin *only* when continuation is genuinely planted — and on the null (continuation = 0) it
is a fair coin. Run:

    python studies/116-power-hour/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from power_hour import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 116 — Power-Hour — synthetic positive control\n")
    print(f"{'continuation':>14} | {'corr':>6} {'t':>6} | "
          f"{'follow bps':>10} {'t':>6} | {'random bps':>10} | follow beats coin?")
    print("-" * 80)
    for cont in (0.00, 0.10, 0.20, 0.30, -0.20):
        panel, _ = data.synthetic_sessions(n_sessions=500, continuation=cont, seed=116)
        corr = st.morning_last_corr(panel)
        follow = st.summarize(st.build_ledger(panel, arm="follow", cost_bps=0), "ret_gross")
        rand = st.summarize(st.build_ledger(panel, arm="random", cost_bps=0, seed=1), "ret_gross")
        beats = "yes" if follow["mean_bps"] > rand["mean_bps"] + 0.05 else "no"
        print(
            f"{cont:>14.2f} | {corr['corr']:>+6.3f} {corr['tstat']:>+6.2f} | "
            f"{follow['mean_bps']:>+10.2f} {follow['tstat']:>+6.2f} | "
            f"{rand['mean_bps']:>+10.2f} | {beats}"
        )

    print("\nThe follow-morning signal is a faithful continuation harvester: it edges the coin")
    print("only when continuation is planted (>0), and is a fair coin at continuation=0.")
    print("On the real 1-hour tape there is no such continuation — see docs/results.md.")


if __name__ == "__main__":
    main()
