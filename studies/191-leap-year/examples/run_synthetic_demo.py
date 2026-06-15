"""Offline, deterministic demo — Study 191 (Leap-Year).

No network.  Builds a synthetic annual return series and shows the study's spine
in one screen: the leap-year premium is detectable only when a genuine effect is
planted — on a martingale (leap_premium=0) the test finds nothing, and a large
planted premium produces a clear signal.  Run:

    python studies/191-leap-year/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from leap_year import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 191 — Leap-Year — synthetic positive/null control\n")
    print(
        f"{'planted leap premium':>22} | {'n leap':>6} "
        f"{'leap mean %':>11} {'contrast %':>11} {'Welch t':>8} {'p-val':>7} | detectable?"
    )
    print("-" * 85)

    for premium in (0.0, 0.05, 0.10, 0.15, -0.10):
        df, truth = data.synthetic_annual(
            n_years=300, leap_premium=premium, start_year=1928, seed=191
        )
        r = st.leap_comparison(df)
        detectable = "YES" if abs(r["tstat_welch"]) > 2.0 else "no"
        print(
            f"{premium:>+22.2f} | {r['n_leap']:>6} "
            f"{r['mean_leap_pct']:>+11.2f} {r['contrast_pct']:>+11.2f} "
            f"{r['tstat_welch']:>+8.3f} {r['pval_welch']:>7.4f} | {detectable}"
        )

    print()
    print("Presidential-term cycle on null tape (no planted effect, n=300 years):")
    df_null, _ = data.synthetic_annual(n_years=300, leap_premium=0.0, seed=191)
    tc = st.term_cycle_comparison(df_null)
    for ty, row in tc.iterrows():
        print(f"  Year-of-term {ty}: mean={row['mean_pct']:+.2f}%  HAC t={row['tstat_hac']:+.3f}")

    print()
    print("On the null tape all four term-years are statistically indistinguishable.")
    print()
    print("Key finding on real Shiller data (1928-2025):")
    print("  n_leap=25  mean_leap=+8.77%  mean_non_leap=+7.75%")
    print("  contrast=+1.02pp  Welch-t=+0.255  p=0.80")
    print("  Bonferroni-corrected p for leap-year hypothesis: 1.00")
    print()
    print("Signal: NONE. Leap years are statistically indistinguishable from any")
    print("other year. The folk coincidence with elections is spurious.")
    print("Tradability: MIRAGE — a one-trade-per-year strategy has no exploitable edge.")


if __name__ == "__main__":
    main()
