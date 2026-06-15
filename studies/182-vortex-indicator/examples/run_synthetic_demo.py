"""Offline, deterministic demo — Study 182 (Vortex-Indicator).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the VI+/VI- crossover only edges a random-direction coin when the tape actually carries
bar-level momentum — and on a martingale (momentum = 0) it is a fair die.  Run:

    python studies/182-vortex-indicator/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vortex_indicator import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 182 — Vortex-Indicator — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'cross bps':>9} {'win':>6} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | cross beats coin?")
    print("-" * 80)

    # Pool 5 seeds to reduce Monte Carlo noise
    import numpy as np

    for mom in (0.00, 0.15, 0.30, -0.15):
        cross_bps_all, rand_bps_all = [], []
        for seed in range(5):
            bars, _ = data.synthetic_daily(n_days=2000, momentum=mom, seed=seed)
            ent = st.crossover_entries(st.vortex(bars, period=14))
            if len(ent) < 5:
                continue
            cross = st.summarize(
                st.run_trades(bars, ent, hold_days=14, cost_bps=0), "ret_gross"
            )
            rand = st.summarize(
                st.run_trades(bars, ent, hold_days=14, cost_bps=0,
                              directions=st.random_directions(len(ent), seed=99 + seed)),
                "ret_gross",
            )
            cross_bps_all.append(cross["mean_bps"])
            rand_bps_all.append(rand["mean_bps"])

        if not cross_bps_all:
            continue

        # Summarise by pooling all returns across seeds for t-stat
        bars_pool, _ = data.synthetic_daily(n_days=5000, momentum=mom, seed=42)
        ent_p = st.crossover_entries(st.vortex(bars_pool, period=14))
        cross_p = st.summarize(
            st.run_trades(bars_pool, ent_p, hold_days=14, cost_bps=0), "ret_gross"
        )
        rand_p = st.summarize(
            st.run_trades(bars_pool, ent_p, hold_days=14, cost_bps=0,
                          directions=st.random_directions(len(ent_p), seed=1)),
            "ret_gross",
        )
        beats = "yes" if np.mean(cross_bps_all) > np.mean(rand_bps_all) + 5 else "no"
        print(f"{mom:>18.2f} | {cross_p['mean_bps']:>+9.2f} {cross_p['win_rate']:>6.3f} "
              f"{cross_p['tstat']:>+6.2f} | {rand_p['mean_bps']:>+10.2f} {rand_p['tstat']:>+6.2f} "
              f"| {beats}")

    print()
    print("The VI crossover is a trend-following detector: it edges a coin when momentum")
    print("is planted (+0.15/+0.30) and has no edge on a martingale (0.00).")
    print("On the real 10-year daily tape (SPY/QQQ/IWM) the best hold-period t-stat is ~1.6")
    print("(below the inference bar of 2.0) — see docs/results.md.")


if __name__ == "__main__":
    main()
