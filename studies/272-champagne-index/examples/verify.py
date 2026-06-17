"""Real-data verification -- Study 272 (Champagne-Index). Regenerates docs/results.md.

Loads the ^GSPC daily price index from the repo-level cache
(``_cache/^GSPC_split_only.parquet``), computes calendar-year price returns, joins
each champagne-shipment year with the *next* year's S&P return (the look-ahead-free
``forward`` convention), and runs the full analysis: predictive OLS with a
Newey-West HAC t-stat, a permutation test, a tercile spread, and a cost-aware
long/short backtest.

The ^GSPC parquet is cache-only (no fetch required -- it ships with the repo).
Run from anywhere:

    python studies/272-champagne-index/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from champagne_index import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 272 -- Champagne-Index -- real-data verification\n")
    print("Data: ^GSPC daily price index (repo-level _cache/^GSPC_split_only.parquet)")
    print("      Champagne shipments: hardcoded in data.py (CIVC figures)\n")

    df = data.fetch_sp500_annual(convention="forward")
    fp = data.fingerprint(df)
    print(f"Data stamp: champagne years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)} forward observations, fingerprint={fp}")
    print("Convention: forward (champagne growth year Y -> S&P return year Y+1)\n")

    s = st.summarize(df, n_permutations=10_000, seed=272)

    print("=== Predictive regression (next-year S&P return on champagne growth) ===")
    print(f"  slope (per 1 sd):   {s['slope']:+.5f}   (folklore predicts < 0)")
    print(f"  correlation:        {s['corr']:+.3f}")
    print(f"  R-squared:          {s['r2']:.3f}")
    print(f"  naive OLS t:        {s['t_ols']:+.3f}")
    print(f"  HAC (NW) t:         {s['t_hac']:+.3f}  (lags={s['lags']})")
    print(f"  Need |t| >= 2 for REAL -> {'PASS' if abs(s['t_hac']) >= 2 else 'FAIL'}\n")

    print("=== Contrarian tercile spread ===")
    print(f"  mean ret after LOW  champagne growth: {s['mean_ret_low_growth_pct']:+.1f}%")
    print(f"  mean ret after HIGH champagne growth: {s['mean_ret_high_growth_pct']:+.1f}%")
    print(f"  spread (low - high):                  {s['spread_pct']:+.1f}pp\n")

    print("=== Permutation tests (10,000 shuffles) ===")
    print(f"  two-sided slope p:   {s['perm_p_slope_two']:.4f}")
    print(f"  one-sided spread p:  {s['perm_p_spread_pos']:.4f}\n")

    print("=== Cost-aware long/short (one-way 10bps, borrow 50bps) ===")
    print(f"  gross annualised:    {s['gross_ann_pct']:+.1f}%")
    print(f"  net annualised:      {s['net_ann_pct']:+.1f}%")
    print(f"  buy & hold (^GSPC):  {s['bah_ann_pct']:+.1f}%  (price-only)")
    print(f"  net vs buy-and-hold: {s['net_ann_pct'] - s['bah_ann_pct']:+.1f}pp/yr\n")

    print("Verdict: Signal=WEAK (corr -0.33, HAC t -1.30 < 2; sign-stable but underpowered)")
    print("         Tradability=MIRAGE (L/S net trails buy-and-hold; shorting fights drift)")


if __name__ == "__main__":
    main()
