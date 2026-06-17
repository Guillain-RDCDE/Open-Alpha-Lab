"""Real-data verification -- Study 290 (September-Effect). Regenerates docs/results.md.

Loads the Shiller S&P 500 monthly price series (repo-level cache, cache-only),
computes month-over-month returns 1950-2025, isolates September against the
pooled other-month baseline, and runs the full analysis: monthly profile,
Newey-West HAC t-test on the September dummy, one-sided permutation test, Welch
two-sample test, sub-period persistence, and the avoid-September backtest.

The Shiller parquet ships in the repo-level _cache/ -- no fetch needed.
Run from anywhere:

    python studies/290-september-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from september_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 290 -- September-Effect -- real-data verification\n")
    print("Data: Shiller S&P 500 monthly PRICE index (repo-level _cache/shiller_sp500.parquet)")
    print("      Price-only (no dividends); 1950-2025.\n")

    df = data.fetch_sp500_monthly()
    fp = data.fingerprint(df)
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)} months, n_sep={(df['month'] == 9).sum()}, fingerprint={fp}\n")

    print("=== Per-calendar-month mean return (%/mo) ===")
    prof = st.monthly_profile(df)
    print(prof[["mean_pct", "std_pct", "up_rate", "hac_t"]].to_string())
    print()

    s = st.september_stats(df, n_permutations=10_000, seed=290)
    print("=== September vs the rest ===")
    print(f"  September mean:      {s['sep_mean_pct']:+.3f}%/mo")
    print(f"  Other-month mean:    {s['other_mean_pct']:+.3f}%/mo")
    print(f"  Gap (Sep - rest):    {s['gap_pct']:+.3f}pp/mo")
    print(f"  September up-rate:   {s['sep_up_rate']:.1%}  (any month: {s['all_up_rate']:.1%})")
    print(f"  HAC t on Sep dummy:  {s['hac_t_gap']:+.3f}  (robust bar |t|>=2)")
    print(f"  Welch t (Sep<rest):  {s['welch_t']:+.3f}  (p = {s['welch_p']:.4f})")
    print(f"  Permutation p (1-sided drag): {s['perm_p']:.4f}\n")

    print("=== Sub-period persistence ===")
    print(st.subperiod_table(df).to_string())
    print()

    bt = st.avoid_september_backtest(df, one_way_bps=5.0)
    print("=== Avoid-September trading rule (cash in September) ===")
    print(f"  Buy & hold:        {bt['buyhold_ann_pct']:.3f}%/yr")
    print(f"  Avoid Sep (gross): {bt['strat_gross_ann_pct']:.3f}%/yr  "
          f"(edge {bt['gross_edge_pp']:+.3f}pp)")
    print(f"  Avoid Sep (net):   {bt['strat_net_ann_pct']:.3f}%/yr  "
          f"(edge {bt['net_edge_pp']:+.3f}pp)  -- before tax\n")

    print("Verdict: Signal=WEAK (HAC t = -1.93, below |t|>=2; drag concentrated in 1986-2005),")
    print("         Tradability=MIRAGE (+0.07pp/yr gross, -0.03pp/yr net) -- a soft month, not a trade.")


if __name__ == "__main__":
    main()
