"""Offline, deterministic demo — Study 164 (Mercury-Retrograde).

No network. Builds a synthetic daily tape and shows the study's spine in one screen:
the retrograde-avoidance strategy only beats buy-and-hold when the tape actually carries
a planted retrograde penalty — on a null tape (no effect) it does not. Run:

    python studies/164-mercury-retrograde/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mercury_retrograde import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 164 — Mercury-Retrograde — synthetic positive control\n")
    print("Retrograde table: {} periods hardcoded (2000–2026)".format(
        len(data.RETROGRADE_PERIODS)
    ))
    retro_days = data.retrograde_dates()
    print(f"Total retrograde calendar days in window: {len(retro_days)}")
    print()

    print(f"{'planted penalty':>18} | {'retro mean bps':>14} {'direct mean bps':>16} "
          f"{'diff bps':>9} {'welch_t':>8} | {'avoid excess CAGR':>18}")
    print("-" * 100)

    for penalty in (0.0, -5.0, -10.0, -20.0):
        ret, mask, truth = data.synthetic_daily(
            n_years=25, retro_penalty_bp=penalty, seed=164
        )
        rtest = st.retrograde_return_test(ret, mask)
        avoid = st.retrograde_avoidance(ret, mask)
        print(
            f"{penalty:>18.1f} | {rtest['retro_mean_bps']:>+14.2f} "
            f"{rtest['direct_mean_bps']:>+16.2f} "
            f"{rtest['diff_bps']:>+9.2f} {rtest['welch_t']:>+8.2f} | "
            f"{avoid['excess_cagr']:>+17.3%}"
        )

    print()
    print("The retrograde detector works: when a -20 bps/day drag is planted,")
    print("the Welch t is large and avoidance generates positive excess CAGR.")
    print("On the null (0.0 bps penalty), there is nothing to find — the verdict")
    print("on the real S&P 500 tape — see docs/results.md.")

    # Show the fraction of trading days that are retrograde
    ret0, mask0, truth0 = data.synthetic_daily(n_years=25, seed=164)
    print(f"\nFraction of trading days that are retrograde: {truth0['frac_retro']:.1%}")
    print(f"(~{truth0['n_retro_days']} retrograde days out of {truth0['n_obs']} total)")


if __name__ == "__main__":
    main()
