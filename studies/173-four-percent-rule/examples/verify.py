"""Real-data verification — Study 173 (Four-Percent-Rule).

Loads the Shiller dataset from the repo-level cache, runs all historical 30-year
rolling cohorts at 4% withdrawal (60/40), computes the safe withdrawal rate at
95% target success, quantifies sequence-of-returns risk, and prints the numbers
that match docs/results.md.

No network is required — the Shiller parquet is pre-staged at _cache/.

    python studies/173-four-percent-rule/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from four_percent_rule import data, strategy as st  # noqa: E402


def main() -> None:
    # ---- Load Shiller real data --------------------------------------------------
    df = data.load_shiller()
    fp = data.fingerprint(df)
    print(f"Shiller real-return tape: {df.index.min()}–{df.index.max()}")
    print(f"Fingerprint: {fp}")
    print(f"Mean equity real return: {df['eq_real_tr'].mean()*100:.2f}%/yr")
    print(f"Mean bond real return:   {df['bond_real_tr'].mean()*100:.2f}%/yr")
    print(f"Equity vol:              {df['eq_real_tr'].std()*100:.1f}%")
    print()

    # ---- Core: 4% rule, 30-year cohorts, 60/40 ----------------------------------
    print("=== 4% Rule — 30-year rolling cohorts (60/40) ===")
    cohorts = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)
    stats   = st.summarize(cohorts)
    print(f"N cohorts:               {stats['n_cohorts']}")
    print(f"Survival rate:           {stats['survival_rate']*100:.1f}%")
    print(f"Median terminal wealth:  {stats['median_terminal_wealth']:.2f}x")
    print(f"Worst terminal wealth:   {stats['worst_terminal_wealth']:.3f}x  (start year: "
          f"{cohorts['terminal_wealth'].idxmin()})")
    print(f"Pct failed:              {stats['pct_fail']:.1f}%")
    print()

    # ---- SWR at 95% and 100% (SAFEMAX) ------------------------------------------
    swr_95   = st.find_swr(df, horizon=30, target_success_rate=0.95)
    swr_100  = st.find_swr(df, horizon=30, target_success_rate=1.00)
    swr_40yr = st.find_swr(df, horizon=40, target_success_rate=0.95)
    print(f"SWR at 95% success (30yr):  {swr_95*100:.2f}%")
    print(f"SAFEMAX / 100% success:     {swr_100*100:.2f}%")
    print(f"SWR at 95% success (40yr):  {swr_40yr*100:.2f}%")
    print()

    # ---- Withdrawal rate sweep --------------------------------------------------
    print("=== Survival by withdrawal rate (30yr, 60/40) ===")
    for wr in (0.03, 0.035, 0.04, 0.045, 0.05, 0.06):
        c = st.run_all_cohorts(df, horizon=30, withdrawal_rate=wr)
        s = st.summarize(c)
        print(f"  WR {wr*100:.1f}%:  survival {s['survival_rate']*100:.1f}%  "
              f"median_tw {s['median_terminal_wealth']:.2f}x")
    print()

    # ---- Sequence-of-returns risk -----------------------------------------------
    print("=== Sequence-of-returns risk ===")
    seq = st.sequence_risk_stats(df, horizon=30, withdrawal_rate=0.04)
    bad  = seq[seq["bad_sequence"]]
    good = seq[~seq["bad_sequence"]]
    print(f"Good sequence (first-5yr equity above median): "
          f"median_tw {good['terminal_wealth'].median():.2f}x")
    print(f"Bad sequence (first-5yr equity below median):  "
          f"median_tw {bad['terminal_wealth'].median():.2f}x")
    print(f"Worst 5 cohorts by terminal wealth:")
    worst5 = seq.nsmallest(5, "terminal_wealth")[["terminal_wealth", "early_eq_return"]]
    for y, row in worst5.iterrows():
        print(f"  {y}: tw={row['terminal_wealth']:.3f}  early_eq={row['early_eq_return']*100:.1f}%/yr")
    print()

    # ---- PE10 quartile analysis -------------------------------------------------
    try:
        raw = pd.read_parquet(data.SHILLER_CACHE)
        raw = raw[raw["SP500"] > 0].copy()
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw["year"] = raw["Date"].dt.year
        pe10 = raw[raw["PE10"] > 0].groupby("year")["PE10"].mean()
        pe10_summary = st.swr_by_pe10_quartile(df, pe10, horizon=30, withdrawal_rate=0.04)
        print("=== Survival by PE10 quartile at retirement date ===")
        print(pe10_summary.to_string())
        print()
    except Exception as e:
        print(f"PE10 analysis skipped: {e}")


if __name__ == "__main__":
    main()
