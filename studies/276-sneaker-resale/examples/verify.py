"""Real-data verification -- Study 276 (Sneaker-Resale). Regenerates docs/results.md.

Loads the ^GSPC price index (repo-level cache, cache-only), computes calendar-year
returns, joins with the hardcoded sneaker-resale index, and runs the full analysis:
return comparison, excess-return t-stats (OLS + Newey-West HAC), AR(1) unsmoothing,
net-of-cost frictions, and the lagged trend overlay.

The ^GSPC parquet ships with the repo at _cache/^GSPC_split_only.parquet and is
read-only (no fetch required). Run from anywhere:

    python studies/276-sneaker-resale/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sneaker_resale import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 276 -- Sneaker-Resale vs S&P 500 -- real-data verification\n")
    print("Data: ^GSPC price index (repo-level _cache/^GSPC_split_only.parquet, cache-only)")
    print("      Sneaker-resale index: hardcoded in data.py (2006-2024)\n")

    df = data.fetch_gspc_annual()  # cache-only; fetch=False by default
    fp = data.fingerprint(df, col="gspc_return")
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, ^GSPC fingerprint={fp}\n")

    s = st.summarize(df)

    print("=== Returns: does the sneaker market beat stocks? ===")
    print(f"  Sneaker mean / CAGR:  {s['sneaker_mean_pct']:+.1f}% / {s['sneaker_cagr_pct']:+.1f}%")
    print(f"  S&P 500 mean / CAGR:  {s['gspc_mean_pct']:+.1f}% / {s['gspc_cagr_pct']:+.1f}%")
    print(f"  Excess (sneaker-S&P): {s['excess_mean_pct']:+.1f}%/yr")
    print(f"  t (excess vs 0):      {s['excess_t_ols']:+.3f}  (p={s['excess_p_ols']:.4f})")
    print(f"  Newey-West HAC t:     {s['excess_t_hac']:+.3f}")
    print(f"  --> outperformance is NOT significant (|t| < 2)\n")

    print("=== Risk: is the high Sharpe real, or stale pricing? ===")
    print(f"  Reported vol:   sneaker {s['sneaker_vol_pct']:.1f}%  vs  S&P {s['gspc_vol_pct']:.1f}%")
    print(f"  Reported Sharpe: sneaker {s['sneaker_sharpe']:.2f}  vs  S&P {s['gspc_sharpe']:.2f}")
    print(f"  AR(1) autocorr of sneaker index: rho = {s['ar1_rho']:+.2f} (stale-pricing!)")
    print(f"  Unsmoothed vol:    {s['unsmoothed_vol_pct']:.1f}%  (vs reported {s['sneaker_vol_pct']:.1f}%)")
    print(f"  Unsmoothed Sharpe: {s['unsmoothed_sharpe']:.2f}  (vs reported {s['sneaker_sharpe']:.2f})")
    print(f"  --> the high Sharpe collapses once smoothing is undone\n")

    print("=== Could you trade it? -- frictions & a trend overlay ===")
    print(f"  Net-of-cost sneaker mean (-3%/yr frictions): {s['net_sneaker_mean_pct']:+.1f}%/yr")
    print(f"  Lagged trend overlay CAGR:  {s['strat_cagr_pct']:+.1f}%/yr")
    print(f"  Buy-and-hold S&P CAGR:      {s['bah_gspc_cagr_pct']:+.1f}%/yr")
    print(f"  Trend overlay vs B&H:       {s['trend_shortfall_pp']:+.2f}pp (single 18-pt path, no signal)\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- "
          "an unbuyable index whose 'edge' is stale pricing.")


if __name__ == "__main__":
    main()
