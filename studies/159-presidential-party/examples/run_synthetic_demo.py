"""Offline, deterministic demo — Study 159 (Presidential-Party).

No network. Builds a synthetic monthly return tape and shows the study's spine:
the D-minus-R gap only appears in the data when a premium is planted, and a
permutation test on the stylised ~17-term sample cannot certify it even then —
exactly mimicking the tiny-n problem that makes the real finding fragile. Run:

    python studies/159-presidential-party/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from presidential_party import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 159 — Presidential-Party — synthetic spine\n")
    print("Testing whether a planted Democrat-administration equity premium is recoverable")
    print("under the same small-n constraints as the real data (~17 terms, ~1160 months).\n")

    print(f"{'premium (bps/mo)':>18} | {'D mean':>9} {'R mean':>9} {'gap':>9} "
          f"{'Welch t':>8} {'perm p':>8} | signal?")
    print("-" * 90)

    for prem in (0.0, 25.0, 50.0, 100.0, 200.0):
        df, _ = data.synthetic_monthly(n_months=1200, dem_premium_bps=prem, seed=159)
        s = st.summarize(df)
        signal = "yes" if abs(s["welch_t"]) > 2.0 and s["perm_pvalue"] < 0.10 else "no"
        print(
            f"{prem:>18.0f} | {s['dem_mean_bps']:>+9.1f} {s['rep_mean_bps']:>+9.1f} "
            f"{s['gap_monthly_bps']:>+9.1f} {s['welch_t']:>+8.2f} {s['perm_pvalue']:>8.3f}"
            f" | {signal}"
        )

    print()
    print("The gap must be enormous (>> actual vol-of-vol) before a Welch t > 2")
    print("and permutation p < 0.10 are achieved simultaneously with only ~17 terms.")
    print()
    print("Real-data headline (Shiller 1927-2023):")
    try:
        df_real = data.load_shiller()
        s_real = st.summarize(df_real)
        print(f"  D mean: {s_real['dem_mean_bps']:+.1f} bps/mo  "
              f"({s_real['dem_mean_ann']:+.1f}%/yr)  |  "
              f"R mean: {s_real['rep_mean_bps']:+.1f} bps/mo  "
              f"({s_real['rep_mean_ann']:+.1f}%/yr)")
        print(f"  Gap: {s_real['gap_monthly_bps']:+.1f} bps/mo  "
              f"({s_real['gap_ann_pct']:+.1f}%/yr)")
        print(f"  Welch t: {s_real['welch_t']:+.2f}  |  "
              f"HAC t (monthly): {s_real['hac_t']:+.2f}  |  "
              f"Permutation p: {s_real['perm_pvalue']:.3f}")
        print(f"  n = {s_real['n_total']} months  |  "
              f"D terms: {s_real['n_terms_dem']}  |  "
              f"R terms: {s_real['n_terms_rep']}")
    except FileNotFoundError as exc:
        print(f"  (Shiller cache not available: {exc})")

    print()
    print("Verdict: the gap looks large in the data, but with only ~17 term-level")
    print("data points it cannot be certified against noise or business-cycle confounds.")
    print("SIGNAL: WEAK  |  TRADABILITY: MIRAGE")


if __name__ == "__main__":
    main()
