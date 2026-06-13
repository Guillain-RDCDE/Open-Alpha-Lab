"""Offline, deterministic demo — Study 105 (Coppock-Curve).

No network. Builds a synthetic monthly tape and shows the study's spine in one screen:
the Coppock buy signal's forward returns vs a random-timing control, on tapes with
varying cycle strength. Run:

    python studies/105-coppock-curve/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coppock_curve import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 105 — Coppock-Curve — synthetic positive control\n")
    print(
        f"{'cycle strength':>16} | {'n sigs':>6} | "
        f"{'cop ann%':>9} {'cop t':>7} | "
        f"{'rand ann%':>9} {'rand t':>7} | "
        f"{'bah ann%':>9} | cop beats rand?"
    )
    print("-" * 90)

    for strength in (0.0, 0.02, 0.04, -0.02):
        bars, _ = data.synthetic_monthly(
            n_months=480,
            trend_strength=strength,
            cycle_period=60,
            seed=105,
        )
        res = st.compare_arms(bars["close"], horizon=12, cost_bps=5.0, rand_seed=42)
        cop = res["coppock"]
        rand = res["random"]
        bah = res["buy_and_hold"]
        beats = "yes" if cop["ann_return_pct"] > rand["ann_return_pct"] + 1.0 else "no"
        print(
            f"{strength:>16.2f} | {res['n_coppock_signals']:>6} | "
            f"{cop['ann_return_pct']:>+9.1f} {cop['tstat']:>+7.2f} | "
            f"{rand['ann_return_pct']:>+9.1f} {rand['tstat']:>+7.2f} | "
            f"{bah['ann_return_pct']:>+9.1f} | {beats}"
        )

    print("\nNotes:")
    print("  - cycle_period=60 months (~5yr), n=480 months (~40 years)")
    print("  - Coppock fires rarely even on long tapes: effective n ~10-20.")
    print("  - At trend_strength=0 the signal has no more timing skill than random.")
    print("  - On the real ^GSPC tape the verdict lives in docs/results.md.")


if __name__ == "__main__":
    main()
