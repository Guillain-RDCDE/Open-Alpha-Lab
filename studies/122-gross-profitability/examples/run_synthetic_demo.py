"""Offline, deterministic demo — Study 122 (Gross-Profitability).

No network. Builds a synthetic (year × ticker) panel with a known GP/A premium and
shows the study's spine in one screen: the high-GP/A quintile only beats the low
quintile when a premium is actually planted — without one, the hedge earns noise.

Run from the repo root:
    python studies/122-gross-profitability/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gross_profitability import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 122 — Gross-Profitability — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'hedge mean':>11} {'hit%':>6} {'t':>6} | "
        f"{'market mean':>12} | beats market?"
    )
    print("-" * 78)

    for premium in (0.00, 0.04, 0.08, -0.04):
        signal, fwd_ret, truth = data.synthetic_panel(
            n_firms=300, n_years=20, gpa_premium=premium, seed=122
        )
        h = st.quintile_hedge(signal, fwd_ret)
        s_hedge = st.summary(h["hedge"])
        s_mkt = st.summary(h["market"])
        beats = "yes" if s_hedge["mean"] > 0.005 else "no"
        print(
            f"{premium:>18.2f} | {s_hedge['mean']:>+11.4f} "
            f"{s_hedge['hit_rate']*100:>6.1f} {s_hedge['tstat']:>+6.2f} | "
            f"{s_mkt['mean']:>+12.4f} | {beats}"
        )

    print()
    print("The hedge only earns meaningfully above zero when the premium is planted.")
    print("On the real tape the result is positive but attenuated by survivorship.")
    print("See docs/results.md for the real-tape verdict.")


if __name__ == "__main__":
    main()
