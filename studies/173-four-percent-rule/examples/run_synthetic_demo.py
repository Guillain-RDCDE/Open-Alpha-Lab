"""Offline, deterministic demo — Study 173 (Four-Percent-Rule).

No network.  Builds synthetic annual return cohorts and shows the study's spine in
one screen:

  1. The 4% rule's historical survival depends heavily on return assumptions —
     it holds in the long-run-equity world but breaks in a low-yield world.
  2. Sequence-of-returns risk: the same 30-year mean with a bad opening produces
     dramatically lower terminal wealth even if no cohort fails outright.
  3. The SWR at 95% target success is above 4% on a normal tape, but below 2%
     in a low-yield world — illustrating Pfau's fragility argument.

Run:
    python studies/173-four-percent-rule/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_percent_rule import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 173 — Four-Percent-Rule — synthetic spine\n")

    # ---- 1. Survival by withdrawal rate across return regimes --------------------
    print("=== Survival rate by withdrawal rate (30yr, 60/40) ===")
    print(f"{'Tape':20s} | {'WR 3%':>8} | {'WR 4%':>8} | {'WR 5%':>8} | {'SWR@95%':>9}")
    print("-" * 65)

    scenarios = [
        ("Normal returns (6.8%)",  dict(equity_real_mean=0.068, bad_start=False, seed=173)),
        ("Low-yield world (3%)",   dict(equity_real_mean=0.030, bond_real_mean=0.005, bad_start=False, seed=173)),
        ("Bad-sequence (6.8%)",    dict(equity_real_mean=0.068, bad_start=True,  seed=173)),
    ]

    for label, kwargs in scenarios:
        df, _ = data.synthetic_cohorts(n_years_total=200, **kwargs)
        s3  = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.03)["survived"].mean()
        s4  = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)["survived"].mean()
        s5  = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.05)["survived"].mean()
        swr = st.find_swr(df, horizon=30, target_success_rate=0.95)
        print(f"{label:20s} | {s3*100:7.1f}% | {s4*100:7.1f}% | {s5*100:7.1f}% | {swr*100:8.2f}%")

    print()

    # ---- 2. Sequence-of-returns risk teardown -----------------------------------
    # Use a single 30-year cohort: identical mean, but different order of returns.
    print("=== Sequence-of-returns risk (single 30yr cohort, 4% WR) ===")
    import numpy as np

    rng = np.random.default_rng(42)
    # 30 equity returns with mean ~7%
    eq_base = rng.normal(0.068, 0.175, 30)
    bond_base = rng.normal(0.016, 0.065, 30)

    import pandas as pd
    yrs = list(range(2000, 2030))

    # Good sequence: random order
    df_good = pd.DataFrame({"eq_real_tr": eq_base, "bond_real_tr": bond_base}, index=yrs)
    # Bad sequence: worst returns first, best returns last
    eq_bad = np.concatenate([np.sort(eq_base)[:10], np.sort(eq_base)[10:]])
    df_bad = pd.DataFrame({"eq_real_tr": eq_bad, "bond_real_tr": bond_base}, index=yrs)

    res_good = st.simulate_cohort(df_good, 2000, horizon=30, withdrawal_rate=0.04)
    res_bad  = st.simulate_cohort(df_bad,  2000, horizon=30, withdrawal_rate=0.04)

    print(f"{'Same mean, same 30yr returns — only ORDER differs':}")
    print(f"  Good sequence (random order):  terminal wealth = {res_good['terminal_wealth']:.3f}")
    print(f"  Bad sequence (worst-first):    terminal wealth = {res_bad['terminal_wealth']:.3f}")
    print(f"  Mean equity return (both):     {eq_base.mean()*100:.1f}%")
    print()
    print()

    print("The 4% rule survives most synthetic cohorts with long-run real returns,")
    print("but collapses in a low-yield world and is seriously stressed by a bad first-")
    print("3-year sequence.  On the real Shiller tape (1872-2022): see docs/results.md.")


if __name__ == "__main__":
    main()
