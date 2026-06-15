"""Offline, deterministic demo — Study 183 (Fisher-Transform).

No network. Builds synthetic daily tapes and shows the study's spine in one screen:

1. The Fisher/Trigger crossover and the raw-x crossover fire on *identical* bars —
   confirming the transform is monotone and adds zero signal information.
2. On a martingale (mean_rev = 0), the Fisher signal is a fair die; it does not beat a
   random-direction control.
3. On a strongly mean-reverting tape, the signal recovers a small edge — confirming the
   engine works when there is something to find.

Run:

    python studies/183-fisher-transform/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fisher_transform import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 183 - Fisher-Transform - synthetic positive control\n")

    # ------------------------------------------------------------------
    # Part 1: monotonicity proof on the martingale tape
    # ------------------------------------------------------------------
    print("=== Part 1: Fisher and raw-x crossovers are identical (monotonicity) ===")
    bars, _ = data.synthetic_daily(n_days=500, mean_rev=0.0, seed=183)
    ft = st.fisher_transform(bars, period=10)
    agree = st.fisher_vs_raw_agreement(ft)
    print(f"  Fisher entries that coincide with raw-x entries: {agree['fisher_in_raw']:.4f}")
    print(f"  Raw-x entries that coincide with Fisher entries:  {agree['raw_in_fisher']:.4f}")
    print("  -> Both should be >=0.99; the transform adds zero signal information.\n")

    # ------------------------------------------------------------------
    # Part 2: signal vs coin across a sweep of mean-reversion strengths
    # ------------------------------------------------------------------
    print(f"{'planted mean_rev':>18} | {'Fisher bps':>10} {'t':>6} | "
          f"{'random bps':>10} {'t':>6} | Fisher beats coin?")
    print("-" * 78)
    for mrev in (-0.20, 0.00, 0.15, 0.30):
        bars, _ = data.synthetic_daily(n_days=800, mean_rev=mrev, seed=183)
        ft = st.fisher_transform(bars, period=10)
        ent = st.crossover_entries(ft)
        fisher = st.summarize(
            st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, ent, hold_days=5, cost_bps=0,
                directions=st.random_directions(len(ent), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if fisher["mean_bps"] > rand["mean_bps"] + 1.0 else "no"
        print(f"{mrev:>18.2f} | {fisher['mean_bps']:>+10.2f} {fisher['tstat']:>+6.2f} | "
              f"{rand['mean_bps']:>+10.2f} {rand['tstat']:>+6.2f} | {beats}")

    print()
    print("The Fisher crossover is a momentum follower (not a mean-reversion signal): it")
    print("beats a coin when momentum (negative mean_rev) is planted, but loses to the coin")
    print("when mean-reversion is planted (positive mean_rev). On a martingale (0.00) it is")
    print("a fair die; the transform itself adds nothing over the raw normalised price.")
    print("On the real daily tape there is no persistent momentum - see docs/results.md.")


if __name__ == "__main__":
    main()
