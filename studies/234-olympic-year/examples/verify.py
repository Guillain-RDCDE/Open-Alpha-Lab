"""Real-data verification — Study 234 (Olympic-Year).  Regenerates docs/results.md numbers.

Loads (or fetches) the yfinance ^GSPC annual series, computes calendar-year returns,
groups by Olympic/non-Olympic year (hardcoded table), and runs the full analysis.

    python studies/234-olympic-year/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from olympic_year import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 234 — Olympic-Year — real ^GSPC data verification\n")

    df = data.load_gspc()
    fp = data.fingerprint(df)
    print(f"^GSPC tape: {df.index.min()}–{df.index.max()}  n={len(df)}  fp={fp}")
    print()

    # Olympic vs non-Olympic group table.
    tbl = st.returns_by_group(df)
    print("=== Olympic vs non-Olympic: mean calendar-year return ===")
    print(f"  {'group':>12} {'mean %':>8} {'std %':>7} {'n':>4}")
    for is_oly, row in tbl.iterrows():
        label = "Olympic" if is_oly else "Non-Olympic"
        print(f"  {label:>12}  {row['mean']:>8.2f}  {row['std']:>7.2f}  {int(row['n']):>4}")
    print()

    # Olympic contrast.
    con = st.olympic_contrast(df)
    print("=== Olympic contrast ===")
    print(f"  Olympic mean:           {con['mean_olympic_pct']:+.2f}%  (n={con['n_olympic']}, "
          f"std={con['std_olympic']:.2f}%)")
    print(f"  Non-Olympic mean:       {con['mean_non_olympic_pct']:+.2f}%  (n={con['n_non_olympic']})")
    print(f"  Grand mean:             {con['grand_mean_pct']:+.2f}%")
    print(f"  Contrast (Oly-NonOly):  {con['contrast_pct']:+.2f} pp")
    print(f"  HAC t-stat:             {con['tstat_hac']:+.4f}")
    print(f"  SE:                     {con['se']:.2f} pp")
    print(f"  [!] n={con['n_olympic']} — low-power warning: {con['low_power_warning']}")
    print()

    # Permutation test.
    print("=== Permutation test (50,000 iterations) ===")
    perm = st.permutation_test(df, n_perm=50_000, seed=234)
    print(f"  Observed contrast:      {perm['observed_contrast_pct']:+.2f} pp")
    print(f"  p-value (two-sided):    {perm['pval_two_sided']:.4f}")
    print(f"  p-value (one-sided):    {perm['pval_one_sided']:.4f}")
    print()

    # Pre/post split.
    split = st.pre_post_split(df, split_year=1972)
    print("=== Pre/post 1972 split (Bretton Woods break) ===")
    print(f"  Pre-1972:  n_oly={split['pre']['n_olympic']}, "
          f"mean_oly={split['pre']['mean_olympic']:+.2f}%, "
          f"mean_non={split['pre']['mean_non']:+.2f}%, "
          f"contrast={split['pre']['contrast']:+.2f} pp")
    print(f"  Post-1972: n_oly={split['post']['n_olympic']}, "
          f"mean_oly={split['post']['mean_olympic']:+.2f}%, "
          f"mean_non={split['post']['mean_non']:+.2f}%, "
          f"contrast={split['post']['contrast']:+.2f} pp")
    print()

    # Timing strategy.
    timing = st.olympic_timing_strategy(df)
    print("=== Olympic timing: hold only in Olympic years ===")
    print(f"  Buy-and-hold annualised:   {timing['bah_ann_pct']:.2f}%/yr")
    print(f"  Olympic-only strategy:     {timing['strat_ann_pct']:.2f}%/yr")
    print(f"  Annual advantage:          {timing['ann_advantage_pct']:+.2f}%/yr")
    print()

    print("=== VERDICT ===")
    print(f"  Signal: NONE — HAC t={con['tstat_hac']:.2f}, permutation p={perm['pval_two_sided']:.4f}.")
    print(f"  Olympic years mean {con['contrast_pct']:+.2f} pp more than non-Olympic,")
    print(f"  entirely within sampling noise at n={con['n_olympic']} observations.")
    print(f"  Tradability: MIRAGE — Olympic-only strategy yields {timing['strat_ann_pct']:.2f}%/yr")
    print(f"  vs {timing['bah_ann_pct']:.2f}%/yr buy-and-hold (you are in cash 74 of 97 years).")
    print(f"  Myth-check: BUSTED — no statistical signal, no mechanism.")


if __name__ == "__main__":
    main()
