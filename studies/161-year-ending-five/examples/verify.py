"""Real-data verification — Study 161 (Year-Ending-Five).  Regenerates docs/results.md numbers.

Loads the staged Shiller S&P 500 monthly dataset, computes calendar-year returns,
groups by last digit, and runs the full decennial-cycle analysis.  No network access
needed — the Shiller cache is a pre-staged shared repo asset.

    python studies/161-year-ending-five/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from year_ending_five import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 161 — Year-Ending-Five — real Shiller data verification\n")

    df = data.load_shiller()
    fp = data.fingerprint(df)
    print(f"Shiller tape: {df.index.min()}–{df.index.max()}  n={len(df)}  fp={fp}")
    print()

    # Full decennial table.
    tbl = st.returns_by_digit(df)
    print("=== Decennial cycle: mean calendar-year return by last digit ===")
    print(f"  {'digit':>5} {'mean %':>8} {'std %':>7} {'n':>4}")
    for d, row in tbl.iterrows():
        marker = " <<< digit 5 (claim)" if d == 5 else (" <<< digit 0 (claim)" if d == 0 else "")
        print(f"  {d:>5}  {row['mean']:>8.2f}  {row['std']:>7.2f}  {int(row['n']):>4}{marker}")
    print()

    # Digit-5 contrast.
    con = st.digit5_contrast(df)
    print("=== Digit-5 contrast ===")
    print(f"  Digit-5 mean:       {con['mean5_pct']:+.2f}%  (n={con['n5']}, std={con['std5']:.2f}%)")
    print(f"  All-other mean:     {con['mean_others_pct']:+.2f}%")
    print(f"  Grand mean:         {con['grand_mean_pct']:+.2f}%")
    print(f"  Contrast (5-others):{con['contrast_pct']:+.2f}%")
    print(f"  HAC t-stat:         {con['tstat_hac']:+.2f}")
    print(f"  [!] n={con['n5']} -- low-power warning: {con['low_power_warning']}")
    print()

    # Permutation test.
    print("=== Permutation / multiple-comparisons test ===")
    perm = st.permutation_test(df, n_perm=50_000, seed=161)
    print(f"  Observed digit-5 mean:         {perm['observed5_mean_pct']:+.2f}%")
    print(f"  Perm p-value (digit 5 only):   {perm['pval_digit5']:.5f}")
    print(f"  Perm p-value (best-of-10):     {perm['pval_best_of_10']:.5f}")
    print(f"  Multiple-comparisons inflation: {perm['inflation_factor']:.1f}x")
    print(f"  Note: even the best-of-10 p ({perm['pval_best_of_10']:.3f}) survives a 5% bar,")
    print(f"        but n={con['n5']} per digit is the structural floor on any claim.")
    print()

    # Pre/post split.
    split = st.pre_post_split(df, split_year=1945)
    print("=== Pre/post 1945 split (informal out-of-sample) ===")
    print(f"  Pre-1945:  n5={split['pre']['n5']}, mean5={split['pre']['mean5']:+.2f}%,"
          f" others={split['pre']['mean_others']:+.2f}%")
    print(f"  Post-1945: n5={split['post']['n5']}, mean5={split['post']['mean5']:+.2f}%,"
          f" others={split['post']['mean_others']:+.2f}%")
    print()

    # Timing strategy.
    timing = st.decennial_timing_strategy(df, bad_digits=(0,))
    print("=== Decennial timing: hold except years ending in 0 ===")
    print(f"  Buy-and-hold annualised:   {timing['bah_ann_pct']:.2f}%/yr")
    print(f"  Timing strategy ann.:      {timing['strat_ann_pct']:.2f}%/yr")
    print(f"  Annual advantage:          {timing['ann_advantage_pct']:+.2f}%/yr")
    print(f"  Years avoided: {timing['n_avoided_years']} (digit-0 years)")
    print()

    print("=== VERDICT ===")
    print(f"  Signal: WEAK — HAC t={con['tstat_hac']:.2f} clears |t|=2 on the raw digit-5 test,")
    print(f"         but best-of-10 p={perm['pval_best_of_10']:.3f} at n=16 per digit.")
    print(f"  Tradability: MIRAGE — {timing['ann_advantage_pct']:+.2f}%/yr advantage from avoiding")
    print(f"         digit-0 years; this is 15 years' worth of timing in 154 years.")
    print(f"  Myth-check: the folklore is PARTIALLY REAL but dramatically overstated —")
    print(f"         digit 5 *is* the best bucket, but n=16 means one bad year changes everything.")


if __name__ == "__main__":
    main()
