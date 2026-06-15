"""Offline, deterministic demo — Study 166 (First-Five-Days).

No network.  Builds a synthetic annual tape and shows the study's spine: the
First-Five-Days Early Warning System only shows a real hit-rate edge when the signal
is planted, and on the null tape (signal_strength = 0) the hit-rate is roughly equal
to the equity-drift base rate — not a fair coin, because markets drift up, but not a
crystal ball either.  Run:

    python studies/166-first-five-days/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from first_five_days import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 166 — First-Five-Days — synthetic F5D sweep\n")
    print(
        f"{'signal_strength':>16} | {'hit_rate':>9} {'base_rate':>9} "
        f"{'pval/coin':>10} {'pval/base':>10} {'contrast%':>10} {'HAC-t':>7} | signal?"
    )
    print("-" * 92)

    for strength in (0.0, 0.10, 0.20, 0.35, 0.50, -0.20):
        df, _ = data.synthetic_annual(n_years=76, signal_strength=strength, seed=166)
        result = st.summarize(df)
        sig = (
            "yes (planted)"
            if strength > 0 and result["pval_vs_coin"] < 0.10
            else "reverse"
            if strength < 0
            else "no"
        )
        print(
            f"{strength:>16.2f} | {result['hit_rate']:>9.3f} {result['base_rate']:>9.3f}"
            f" {result['pval_vs_coin']:>10.4f} {result['pval_vs_base_rate']:>10.4f}"
            f" {result['contrast_pct']:>10.2f} {result['tstat_hac']:>7.2f} | {sig}"
        )

    print()
    print("Key insight: on the null tape, hit-rate > 50% emerges from equity drift")
    print("(markets go up most years regardless of the first five days), NOT from any")
    print("predictive signal.  The pval_vs_base_rate test exposes this: does F5D beat")
    print("*knowing markets trend up*?  On 76 real years: p = 0.82 -- no, it does not.")
    print()
    print("Full-sample HAC t-stat looks impressive at ~4.9, but:")
    print("  - post-1985 sub-period: t = 2.9, hit-rate = 60% (not sig vs coin, p=0.13)")
    print("  - the F5D return is PART of the full-year return (5 days of overlap),")
    print("    which creates a mechanical correlation that inflates the t-stat.")
    print("  - strategy (long if up-F5D, flat otherwise) barely beats buy-and-hold.")
    print("See docs/results.md for the full real-tape run.")


if __name__ == "__main__":
    main()
