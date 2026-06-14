"""Offline, deterministic demo — Study 126 (Parabolic-SAR).

No network.  Builds synthetic daily tapes and shows the study's spine in one screen:
Wilder's Parabolic SAR only beats a random-direction coin when the tape actually carries
directional momentum — on a martingale (momentum = 0) it is a fair die.  Run:

    python studies/126-parabolic-sar/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parabolic_sar import data, strategy as st  # noqa: E402


def _edge(bars, seed: int = 1) -> tuple[dict, dict]:
    ent = st.flip_entries(bars)
    sar_s = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
    rand_s = st.summarize(
        st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                      directions=st.random_directions(len(ent), seed=seed)),
        "ret_gross",
    )
    return sar_s, rand_s


def main() -> None:
    print("Study 126 — Parabolic-SAR — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'flips':>6} | {'SAR bps':>8} {'win':>6} {'t':>7} | "
          f"{'coin bps':>8} {'t':>7} | SAR beats coin?")
    print("-" * 85)

    for mom in (0.00, 0.10, 0.20, 0.30, -0.10):
        bars, _ = data.synthetic_daily(n_days=600, momentum=mom, seed=126)
        ent = st.flip_entries(bars)
        sar_s, rand_s = _edge(bars)
        beats = "yes" if sar_s["mean_bps"] > rand_s["mean_bps"] + 0.05 else "no "
        print(f"{mom:>18.2f} | {len(ent):>6d} | {sar_s['mean_bps']:>+8.2f} "
              f"{sar_s['win_rate']:>6.3f} {sar_s['tstat']:>+6.2f} | "
              f"{rand_s['mean_bps']:>+8.2f} {rand_s['tstat']:>+6.2f} | {beats}")

    print()
    print("The SAR is a faithful momentum harvester: it only edges the coin when")
    print("momentum is planted.  On a martingale (0.00) it is a fair coin.")
    print("On the real daily tape there is no robust trend signal — see docs/results.md.")


if __name__ == "__main__":
    main()
