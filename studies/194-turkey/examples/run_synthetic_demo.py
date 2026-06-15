"""Offline, deterministic demo — Study 194 (Turkey).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the Thanksgiving pre/post-holiday anomaly is detectable only when a genuine effect is
planted — on a martingale (both effects=0) the Thanksgiving Wednesdays and Black
Fridays are statistically indistinguishable from any other day. Run:

    python studies/194-turkey/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turkey import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 194 — Turkey — synthetic positive/negative control")
    print("Thanksgiving pre/post-holiday effect (^GSPC, 75 synthetic years)\n")

    # --- Wednesday before Thanksgiving ---
    print("A) Wednesday before Thanksgiving vs other Wednesdays")
    print(
        f"{'wed_effect':>12} | {'n_wed':>5} {'mean_wed(bps)':>14} "
        f"{'contrast(bps)':>14} {'Welch t':>8} {'p-val':>7} | detectable?"
    )
    print("-" * 80)
    for eff in (0.0, -2.0, +2.0, +4.0):
        df, truth = data.synthetic_daily(n_years=75, wed_effect=eff, fri_effect=0.0, seed=194)
        r = st.wednesday_comparison(df)
        detectable = "YES" if abs(r["tstat_welch"]) > 2.0 else "no"
        print(
            f"{eff:>12.1f} | {r['n_wed']:>5} {r['mean_wed_bps']:>+14.2f} "
            f"{r['contrast_bps']:>+14.2f} {r['tstat_welch']:>+8.3f} "
            f"{r['pval_welch']:>7.4f} | {detectable}"
        )

    print()
    # --- Friday after Thanksgiving (Black Friday) ---
    print("B) Friday after Thanksgiving (Black Friday) vs other Fridays")
    print(
        f"{'fri_effect':>12} | {'n_fri':>5} {'mean_fri(bps)':>14} "
        f"{'contrast(bps)':>14} {'Welch t':>8} {'p-val':>7} | detectable?"
    )
    print("-" * 80)
    for eff in (0.0, -2.0, +2.0, +4.0):
        df, truth = data.synthetic_daily(n_years=75, wed_effect=0.0, fri_effect=eff, seed=194)
        r = st.friday_comparison(df)
        detectable = "YES" if abs(r["tstat_welch"]) > 2.0 else "no"
        print(
            f"{eff:>12.1f} | {r['n_fri']:>5} {r['mean_fri_bps']:>+14.2f} "
            f"{r['contrast_bps']:>+14.2f} {r['tstat_welch']:>+8.3f} "
            f"{r['pval_welch']:>7.4f} | {detectable}"
        )

    print()
    print("The engine detects planted effects but finds nothing when the tape is fair.")
    print("Real ^GSPC (1928-2026): Wed contrast ~ -1 bps (p~0.91), Fri contrast ~+16 bps (p~0.10).")
    print("Signal: WEAK at best. Black Friday HAC t=2.01 but Bonferroni p=0.20. Tradability: MIRAGE.")


if __name__ == "__main__":
    main()
