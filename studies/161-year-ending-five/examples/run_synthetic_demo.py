"""Offline, deterministic demo — Study 161 (Year-Ending-Five).

No network.  Builds a synthetic 150-year annual return tape and demonstrates
the study's spine in one screen:

- On the NULL tape (digit5_boost=0) the decennial grouping finds noise.
- On the BOOSTED tape (digit5_boost=0.20) the digit-5 years clearly outperform.
- The permutation multiple-comparisons correction shows how much the
  single-hypothesis p inflates when we're testing all 10 digits post-hoc.

Run:
    python studies/161-year-ending-five/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from year_ending_five import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 161 — Year-Ending-Five — synthetic spine demo\n")

    for boost, label in [(0.0, "NULL (no boost)"), (0.20, "BOOSTED (+20pp log digit-5)")]:
        df, truth = data.synthetic_annual(n_years=150, digit5_boost=boost, seed=161)
        con = st.digit5_contrast(df)
        perm = st.permutation_test(df, n_perm=5_000, seed=0)

        print(f"Tape: {label}")
        print(f"  n_years={truth['n_years']}, n_digit5={truth['n_digit5_years']}")
        print(f"  Grand mean:     {con['grand_mean_pct']:+.2f}%")
        print(f"  Digit-5 mean:   {con['mean5_pct']:+.2f}%  (std {con['std5']:.2f}%)")
        print(f"  Contrast vs others: {con['contrast_pct']:+.2f}%")
        print(f"  HAC t-stat:     {con['tstat_hac']:+.2f}")
        print(f"  Perm p (digit-5 only):  {perm['pval_digit5']:.4f}")
        print(f"  Perm p (best-of-10):    {perm['pval_best_of_10']:.4f}  "
              f"  [~{perm['inflation_factor']:.1f}x inflation from 10 simultaneous tests]")
        print()

    # Full decennial table on the null tape (to show what 'noise' looks like).
    df_null, _ = data.synthetic_annual(n_years=150, digit5_boost=0.0, seed=161)
    tbl = st.returns_by_digit(df_null)
    print("Null-tape decennial table (digit -> mean%):")
    print(f"  {'digit':>5} {'mean %':>8} {'std %':>7} {'n':>4}")
    for d, row in tbl.iterrows():
        marker = " <<< digit 5" if d == 5 else ""
        print(f"  {d:>5}  {row['mean']:>8.2f}  {row['std']:>7.2f}  {int(row['n']):>4}{marker}")

    print(
        "\nOn a null tape the decennial table produces one 'best' digit by chance every run."
        "\nThe multiple-comparisons correction is essential.  On the real Shiller tape,"
        "\ndigit 5 clears the Bonferroni bar -- but n=16 demands humility."
        "\nSee docs/results.md and the notebooks for the real-data findings."
    )


if __name__ == "__main__":
    main()
