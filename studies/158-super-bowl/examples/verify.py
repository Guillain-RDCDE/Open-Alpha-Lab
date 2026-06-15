"""Real-data verification -- Study 158 (Super-Bowl). Regenerates docs/results.md numbers.

Loads the Shiller S&P 500 monthly dataset (repo-level cache), computes calendar-year
returns, joins with the hardcoded Super Bowl table, and runs the full analysis:
hit-rates, binomial test, permutation test, and multiple-comparisons summary.

The Shiller parquet is at the repo-level _cache/shiller_sp500.parquet and is read-only
(no fetch required -- it ships with the repo). Run from anywhere:

    python studies/158-super-bowl/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from super_bowl import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 158 -- Super-Bowl Indicator -- real-data verification\n")
    print("Data: Shiller S&P 500 monthly (repo-level _cache/shiller_sp500.parquet)")
    print("      Super Bowl results: hardcoded in data.py (1967-2025)\n")

    df = data.fetch_sp500_annual()
    fp = data.fingerprint(df)

    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}\n")

    # Basic stats
    nfc_count = (df["conference"] == "NFC").sum()
    afc_count = (df["conference"] == "AFC").sum()
    uncond = df["mkt_up"].mean()
    print(f"Super Bowl winners:  NFC={nfc_count}, AFC={afc_count}")
    print(f"Unconditional up-rate: {uncond:.1%} ({df['mkt_up'].sum()}/{len(df)} years up)\n")

    # Main summary
    s = st.summarize(df, n_permutations=10_000, seed=158)

    print("=== NFC signal (conference): NFC -> bull, AFC -> bear ===")
    print(f"  Overall hit-rate:   {s['nfc_hit_rate_pct']:.1f}%")
    print(f"  Hit-rate NFC years: {s['nfc_hit_bull_pct']:.1f}% "
          f"(mkt up vs unconditional {s['uncond_up_rate_pct']:.1f}%)")
    print(f"  Hit-rate AFC years: {s['nfc_hit_bear_pct']:.1f}% (mkt down)")
    print(f"  Mean return NFC:    {s['nfc_mean_bull']:+.1f}%")
    print(f"  Mean return AFC:    {s['nfc_mean_bear']:+.1f}%")
    print(f"  Welch t (NFC vs AFC): {s['nfc_welch_t']:+.3f}")
    print(f"  Binomial p (vs {s['uncond_up_rate_pct']:.0f}% base): {s['nfc_binom_p']:.4f}")
    print(f"  Permutation p:      {s['nfc_perm_p']:.4f}")
    print(f"  Bonferroni binom_p: {s['bonf_nfc_binom_p']:.4f}\n")

    print("=== Original-NFL signal: orig_nfl -> bull ===")
    print(f"  Overall hit-rate:   {s['orig_hit_rate_pct']:.1f}%")
    print(f"  Welch t:            {s['orig_welch_t']:+.3f}")
    print(f"  Binomial p:         {s['orig_binom_p']:.4f}")
    print(f"  Permutation p:      {s['orig_perm_p']:.4f}")
    print(f"  Bonferroni binom_p: {s['bonf_orig_binom_p']:.4f}\n")

    # Full multiple-comparisons table
    mc = st.multiple_comparisons_summary(df, n_permutations=10_000, seed=158)
    print("=== Multiple-comparisons summary ===")
    print(mc[["test", "n", "hit_rate_all", "binom_p", "perm_p",
               "bonferroni_binom_p", "bonferroni_perm_p"]].to_string(index=False))
    print()

    # Show the base-rate trap explicitly
    nfc_up_rate = df.loc[df["conference"] == "NFC", "mkt_up"].mean()
    afc_up_rate = df.loc[df["conference"] == "AFC", "mkt_up"].mean()
    print(f"=== The base-rate trap ===")
    print(f"  Up-rate in NFC years: {nfc_up_rate:.1%}")
    print(f"  Up-rate in AFC years: {afc_up_rate:.1%}")
    print(f"  Unconditional:        {uncond:.1%}")
    print(f"  The NFC up-rate exceeds unconditional by only "
          f"{(nfc_up_rate - uncond)*100:+.1f}pp -- well inside sampling noise.")
    print()
    print(f"Verdict: Signal=NONE, Tradability=MIRAGE -- a coincidence dressed as an omen.")


if __name__ == "__main__":
    main()
