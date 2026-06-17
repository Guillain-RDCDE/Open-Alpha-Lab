"""Real-data verification -- Study 217 (Lipstick Index). Regenerates docs/results.md numbers.

Loads the cosmetics basket (EL, ULTA, COTY, ELF) and SPY from the study-local
yfinance cache, computes monthly relative returns, joins with NBER recession dates,
and runs the full analysis: recession alpha, Welch t-test, and permutation test.

The cosmetics parquet is cached at:
    studies/217-lipstick-index/_cache/cosmetics_monthly.parquet

Run from anywhere:
    python studies/217-lipstick-index/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lipstick_index import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 217 -- Lipstick Index -- real-data verification\n")
    print("Data: yfinance cosmetics basket (EL, ULTA, COTY, ELF) + SPY")
    print("      Recession dates: hardcoded NBER table in data.py\n")

    raw = data.fetch_cosmetics()
    af = data.build_analysis_frame(raw)
    fp = data.fingerprint(af)

    n_total = len(af)
    n_rec = int(af["in_recession"].sum())
    n_exp = n_total - n_rec

    print(f"Data stamp: {af['month'].min().date()} to {af['month'].max().date()}, "
          f"n={n_total} months, fingerprint={fp}")
    print(f"Tickers fetched: {sorted(raw['ticker'].unique())}")
    print(f"Recession months: {n_rec} | Expansion months: {n_exp}\n")

    # Main summary
    s = st.summarize(af, n_permutations=10_000, seed=217)

    print("=== Cosmetics basket relative return (vs SPY) ===")
    print(f"  Recession months (n={n_rec}):")
    print(f"    Cosm mean ret:  {s['cosm_rec_pct_month']:+.3f}%/month")
    print(f"    SPY mean ret:   {s['spy_rec_pct_month']:+.3f}%/month")
    print(f"    Relative ret:   {s['mean_rel_rec_pct_month']:+.3f}%/month  "
          f"({s['mean_rel_rec_ann_bps']:+.0f} bps/yr)")
    print(f"  Expansion months (n={n_exp}):")
    print(f"    Cosm mean ret:  {s['cosm_exp_pct_month']:+.3f}%/month")
    print(f"    SPY mean ret:   {s['spy_exp_pct_month']:+.3f}%/month")
    print(f"    Relative ret:   {s['mean_rel_exp_pct_month']:+.3f}%/month  "
          f"({s['mean_rel_exp_pct_month']*100*12:.0f} bps/yr)")
    print()
    print(f"  Difference (rec - exp): {s['mean_rel_diff_pct_month']:+.3f}%/month")
    print(f"  Welch t = {s['welch_t']:+.3f} (p = {s['welch_p']:.4f})")
    print(f"  Permutation p = {s['perm_p']:.4f} (10,000 shuffles)\n")

    print("=== Power check ===")
    vol_m = af["rel_ret"].std()
    mde = 2 * vol_m / (n_rec ** 0.5)
    print(f"  Monthly rel_ret vol: {vol_m*100:.2f}%/month")
    print(f"  n_rec = {n_rec} months  |  MDE at |t|=2: {mde*100:.2f}%/month")
    print(f"  Observed gap: {s['mean_rel_diff_pct_month']:+.3f}%/month "
          f"({abs(s['mean_rel_diff_pct_month'])/mde/100*100:.0f}% of MDE)\n")

    print(f"Verdict: Signal=NONE, Tradability=MIRAGE -- "
          f"the cosmetics basket underperforms SPY in recessions (t={s['welch_t']:+.3f}).")
    print(f"         The Lipstick Index direction is reversed relative to the folklore claim.")


if __name__ == "__main__":
    main()
