"""Real-data verification -- Study 275 (Whisky-Cask). Regenerates docs/results.md numbers.

Loads the S&P 500 ^GSPC daily close (repo-level cache), computes calendar-year price
returns, joins with the hardcoded annual whisky index, and runs the full analysis:
raw performance, Geltner unsmoothing, the cost wedge, and the HAC excess-return test.

Cache-only (fetch=False): the ^GSPC parquet ships in the repo-level _cache/. Run from
anywhere:

    python studies/275-whisky-cask/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from whisky_cask import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 275 -- Whisky-Cask -- real-data verification\n")
    print("Data: S&P 500 ^GSPC daily close (repo-level _cache/^GSPC_split_only.parquet)")
    print("      Whisky index: hardcoded appraisal-based annual index in data.py\n")

    df = data.fetch_gspc_annual(fetch=False)
    fp = data.fingerprint(df)
    print(f"Data stamp: years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}\n")

    perf = st.performance_stats(df)
    print("=== Raw performance (price-only both sides) ===")
    print(f"  Whisky annualised:  {perf['whisky_ann']:+.2f}%   vol {perf['whisky_vol']:.2f}%   "
          f"Sharpe {perf['whisky_sharpe']:.3f}")
    print(f"  S&P 500 annualised: {perf['sp_ann']:+.2f}%   vol {perf['sp_vol']:.2f}%   "
          f"Sharpe {perf['sp_sharpe']:.3f}")
    print(f"  Correlation: {perf['corr']:+.3f}   |  whisky beats S&P in "
          f"{perf['whisky_beats_frac']:.0%} of years\n")

    sm = st.smoothing_stats(df)
    print("=== Geltner unsmoothing (the killer) ===")
    print(f"  lag-1 autocorrelation: {sm['rho_lag1']:.3f}")
    print(f"  vol: {sm['raw_vol_pct']:.2f}% -> {sm['unsmoothed_vol_pct']:.2f}% "
          f"({sm['vol_inflation_x']:.2f}x)")
    print(f"  Sharpe: {sm['raw_sharpe']:.3f} -> {sm['unsmoothed_sharpe']:.3f}\n")

    cost = st.net_of_costs(df)
    print("=== The cost wedge ===")
    print(f"  gross {cost['gross_ann_pct']:.2f}%/yr -> net {cost['net_ann_pct']:.2f}%/yr "
          f"(wedge {cost['total_cost_wedge_pct']:.2f}pp/yr)\n")

    test = st.excess_return_test(df, n_lags=1)
    print("=== HAC excess-return test (whisky - S&P) ===")
    print(f"  mean annual excess: {test['mean_excess_pct']:+.2f}pp")
    print(f"  HAC (Newey-West, 1 lag) t-stat: {test['hac_t']:+.3f}")
    print(f"  whisky-wins fraction: {test['win_frac']:.0%}\n")

    print("Verdict: Signal=NONE (excess t < 2, Sharpe is a smoothing artifact),")
    print("         Tradability=MIRAGE (no vehicle; ~6pp/yr cost wedge) -- "
          "an appraisal mirage in a bottle.")


if __name__ == "__main__":
    main()
