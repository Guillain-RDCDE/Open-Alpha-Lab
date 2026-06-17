"""Real-data verification -- Study 270 (Underwear-Index). Regenerates docs/results.md numbers.

Loads the Shiller S&P 500 monthly dataset (repo-level cache), joins with the hardcoded
underwear index and the NBER recession calendar, and runs the full analysis: the
leading-indicator Fisher exact + permutation tests, the coincident-view tautology check,
and the no-look-ahead timing backtest with a HAC t-stat.

Cache-only by default (the Shiller parquet ships with the repo). Run from anywhere:

    python studies/270-underwear-index/examples/verify.py          # cache-only
    python studies/270-underwear-index/examples/verify.py --fetch  # pull ^GSPC from yfinance
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from underwear_index import data, strategy as st  # noqa: E402


def main(fetch: bool = False) -> None:
    print("Study 270 -- Men's Underwear Index (Greenspan) -- real-data verification\n")
    print("Data: Shiller S&P 500 monthly (repo-level _cache/shiller_sp500.parquet)")
    print("      Underwear index + NBER recessions: hardcoded in data.py (1992-2024)")
    print("      NOTE: the underwear series is a curated reconstruction, not audited.\n")

    df = data.fetch_sp500_annual(fetch=fetch)
    fp = data.fingerprint(df)
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}")
    print(f"Dip years: {list(df.loc[df['dip'],'year'])}")
    print(f"NBER recession years: {sorted(data.US_RECESSIONS)}\n")

    s = st.summarize(df, n_permutations=10_000, seed=270)

    print("=== Leading-indicator view (dip in Y -> recession in Y+1) -- the genuine claim ===")
    print(f"  Unconditional next-year recession rate: {s['uncond_rec_pct']:.1f}%")
    print(f"  Recession rate after a DIP:             {s['rec_rate_dip_pct']:.1f}%  (n_dips={s['n_dips']})")
    print(f"  Recession rate after NO dip:            {s['rec_rate_nodip_pct']:.1f}%")
    print(f"  Fisher exact p (one-sided):             {s['next_fisher_p']:.4f}")
    print(f"  Permutation p:                          {s['next_perm_p']:.4f}\n")

    print("=== Coincident view (same-year) -- TAUTOLOGICAL in a reconstructed series ===")
    print(f"  Recession rate in dip years:            {s['same_rec_rate_dip_pct']:.1f}%")
    print(f"  Fisher exact p:                         {s['same_fisher_p']:.4f}  <- built-in, not a finding")
    print(f"  Permutation p:                          {s['same_perm_p']:.4f}\n")

    print("=== Tradable rule (dip -> flat next year), no look-ahead, 10bps one-way ===")
    print(f"  Buy & hold S&P:                         {s['bh_ann_pct']:.1f}%/yr")
    print(f"  Underwear timing (net):                 {s['strat_net_pct']:.1f}%/yr")
    print(f"  Turnover:                               {s['turnover']:.2f}")
    print(f"  HAC (Newey-West) t on the spread:       {s['nw_t']:+.2f}\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a vivid story, not a predictor.")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
