"""Offline, deterministic demo — Study 150 (SAD-Effect).

No network. Builds a synthetic monthly return tape and shows the study's spine in
one screen: the SAD seasonal signal is detectable only when a premium is planted.
On a null tape (sad_premium_bp = 0), the onset/recovery split is noise. Run:

    python studies/150-sad-effect/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sad_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 150 — SAD-Effect — synthetic positive control\n")
    print("Does the SAD seasonal strategy beat buy-and-hold? Only when we plant the signal.\n")

    print(f"{'planted premium (bp)':>22} | "
          f"{'onset t':>8} {'recovery t':>11} | "
          f"{'dl_slope_t':>11} | "
          f"{'strat CAGR%':>12} {'BaH CAGR%':>10}")
    print("-" * 85)

    for premium in (0.0, 20.0, 40.0, -20.0):
        ret, truth = data.synthetic_monthly(n_years=80, sad_premium_bp=premium, seed=150)
        split = st.seasonal_split(ret)
        dl = data.daylight_proxy(ret.index)
        reg = st.daylight_regression(ret, dl)
        strat = st.sad_strategy(ret)
        bah = st.buy_hold(ret)
        s_s = st.summary(strat)
        s_b = st.summary(bah)
        print(
            f"{premium:>22.1f} | "
            f"{split['onset_t_vs_unconditional']:>+8.2f} "
            f"{split['recovery_t_vs_unconditional']:>+11.2f} | "
            f"{reg['slope_tstat_hac']:>+11.2f} | "
            f"{s_s['cagr']*100:>12.1f} {s_b['cagr']*100:>10.1f}"
        )

    print()
    print("With no planted signal (premium = 0): onset/recovery t-stats are noise,")
    print("daylight slope is ~zero, and the strategy underperforms buy-and-hold (misses 8/12 months).")
    print("With a large planted signal (premium = 40): the machinery finds it.")
    print()
    print("On the real Shiller tape (1871–2026): onset t = -1.31, recovery t = +0.68,")
    print("daylight slope t = 0.16 — essentially null. See docs/results.md.")


if __name__ == "__main__":
    main()
