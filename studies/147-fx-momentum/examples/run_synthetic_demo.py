"""Offline, deterministic demo — Study 147 (FX-Momentum).

No network. Builds a synthetic G10 FX monthly return panel and shows the study's
spine in one screen: the 12-1 momentum sort (long top-3, short bottom-3 G10
currencies) only beats a random-ranking control when cross-sectional return
persistence is planted in the data — and on a null panel (i.i.d. returns) it
adds nothing over flipping a coin on ranks.

Run from the repo root:

    python studies/147-fx-momentum/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_momentum import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 147 — FX-Momentum — synthetic positive control\n")
    print(f"{'momentum_signal':>16} | {'mom ann%':>9} {'mom t':>7} | "
          f"{'rand ann%':>10} {'rand t':>7} | beats random?")
    print("-" * 75)

    for sig in (0.00, 0.005, 0.010, 0.015, 0.020):
        returns, _ = data.synthetic_panel(n_months=240, momentum_signal=sig, seed=147)
        ranks_m = st.momentum_ranks(returns)
        ranks_r = st.random_ranks(returns, seed=1)

        led_m = st.build_portfolio(returns, ranks_m, cost_bps=0)
        led_r = st.build_portfolio(returns, ranks_r, cost_bps=0)

        s_m = st.summarize(led_m, "ret_gross")
        s_r = st.summarize(led_r, "ret_gross")

        beats = "yes" if s_m["mean_ann_pct"] > s_r["mean_ann_pct"] + 0.05 else "no"
        print(f"{sig:>16.3f} | {s_m['mean_ann_pct']:>+9.2f} {s_m['tstat']:>+7.2f} | "
              f"{s_r['mean_ann_pct']:>+10.2f} {s_r['tstat']:>+7.2f} | {beats}")

    print()
    print("The momentum sort is a faithful cross-sectional persistence harvester:")
    print("it outperforms random ranking only when persistence is planted.")
    print("On the real G10 tape the signal is weak and decayed — see docs/results.md.")


if __name__ == "__main__":
    main()
