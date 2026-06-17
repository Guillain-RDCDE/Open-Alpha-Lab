"""Real-data verification -- Study 266 (Misery-Index). Regenerates docs/results.md.

Loads the December CPI-YoY + unemployment table (hardcoded in data.py), joins
the December misery print of year Y to the S&P 500 calendar-year price return of
year Y+1 (from the repo-level cache `_cache/^GSPC_split_only.parquet`), and runs
the full analysis: HAC predictive regressions, the tercile sort, and the
contrarian timing backtest.

Cache-only (no fetch required -- the parquet ships with the repo). Run from
anywhere:

    python studies/266-misery-index/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from misery_index import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 266 -- Misery-Index -- real-data verification\n")
    print("Data: misery table hardcoded in data.py (1948-2025)")
    print("      S&P 500 Dec/Dec price returns from _cache/^GSPC_split_only.parquet\n")

    df = data.fetch_sp500_annual()
    fp = data.fingerprint(df)

    print(f"Data stamp: misery-years {df['year'].min()}-{df['year'].max()} "
          f"-> return-years {df['year'].min()+1}-{df['year'].max()+1}, "
          f"n={len(df)}, fingerprint={fp}\n")

    uncond = df["mkt_up"].mean()
    print(f"Unconditional S&P up-rate: {uncond:.1%} "
          f"({int(df['mkt_up'].sum())}/{len(df)} forward years up)\n")

    s = st.summarize(df)

    print("=== Predictive regressions (per 1-SD, HAC 3-lag) ===")
    print(f"  Misery level : beta = {s['level_beta_pp']:+.2f}pp  "
          f"OLS t = {s['level_t_ols']:+.3f}  HAC t = {s['level_t_hac']:+.3f}  "
          f"R2 = {s['level_r2']:.4f}")
    print(f"  Misery change: beta = {s['chg_beta_pp']:+.2f}pp  "
          f"OLS t = {s['chg_t_ols']:+.3f}  HAC t = {s['chg_t_hac']:+.3f}  "
          f"R2 = {s['chg_r2']:.4f}\n")

    print("=== High-minus-low tercile sort ===")
    print(f"  Low-misery mean fwd return : {s['low_mean_pp']:+.1f}%  "
          f"(up-rate {s['low_up_pct']:.1f}%)")
    print(f"  Mid-misery mean fwd return : {s['mid_mean_pp']:+.1f}%")
    print(f"  High-misery mean fwd return: {s['high_mean_pp']:+.1f}%  "
          f"(up-rate {s['high_up_pct']:.1f}%)")
    print(f"  Spread (high - low)        : {s['spread_pp']:+.1f}pp  "
          f"Welch t = {s['terc_welch_t']:+.3f}, p = {s['terc_welch_p']:.4f}\n")

    print("=== Contrarian timing backtest (long S&P after high-misery print) ===")
    print(f"  Fraction of years in market : {s['frac_in_market']:.1%}")
    print(f"  Turnover                    : {s['turnover_per_yr']:.3f} switches/yr")
    print(f"  Buy & hold CAGR             : {s['bah_cagr_pct']:+.1f}%/yr")
    print(f"  Misery-timing gross CAGR    : {s['gross_cagr_pct']:+.1f}%/yr")
    print(f"  Misery-timing net CAGR      : {s['net_cagr_pct']:+.1f}%/yr")
    print(f"  Net minus buy-and-hold      : {s['net_minus_bah_pp']:+.1f}pp/yr\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- a welfare gauge, not a market timer.")


if __name__ == "__main__":
    main()
