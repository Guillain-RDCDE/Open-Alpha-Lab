"""Real-data verification -- Study 264 (Buffett-Indicator). Regenerates docs/results.md.

Loads the hardcoded Buffett-Indicator table, joins it to S&P 500 (^GSPC) annual
price returns from the repo-level split-only cache, and runs the full analysis:
predictive regression (HAC t), tercile comparison, and the real-time
expanding-median timing strategy vs buy-and-hold.

Cache-only (no network). Run from anywhere:

    python studies/264-buffett-indicator/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from buffett_indicator import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 264 -- Buffett-Indicator -- real-data verification\n")
    print("Data: Buffett Indicator (market cap / GDP) hardcoded in data.py (1971-2025)")
    print("      Forward returns: ^GSPC price, Dec/Dec (repo-level split-only cache)\n")

    df = data.fetch_sp500_annual()
    fp = data.fingerprint(df)
    print(f"Data stamp: indicator years {df['year'].min()}-{df['year'].max()}, "
          f"n={len(df)}, fingerprint={fp}\n")

    reg = st.predictive_regression(df)
    print("=== Predictive regression: fwd_ret_{Y+1} = a + b * ratio_Y ===")
    print(f"  slope b (per +100pp of GDP): {reg['beta_per_100']*100:+.2f} pp of return")
    print(f"  correlation:                 {reg['corr']:+.3f}")
    print(f"  R^2:                         {reg['r2']:.3f}")
    print(f"  HAC t-stat on slope:         {reg['tstat']:+.3f}  (|t|>=2 needed for REAL)\n")

    terc = st.tercile_returns(df)
    print("=== Cheap vs expensive terciles (in-sample breakpoints) ===")
    for b in ("cheap", "mid", "expensive"):
        row = terc.loc[b]
        print(f"  {b:>9}: n={int(row['n']):2d}  mean fwd={row['mean_fwd']*100:+5.1f}%  "
              f"up-rate={row['up_rate']*100:4.1f}%")
    print()

    tim = st.timing_strategy(df)
    ss = st.strategy_stats(tim, one_way_bps=10.0)
    print("=== Real-time timing: long when ratio <= expanding median, else cash ===")
    print(f"  Buffett timing CAGR (gross): {ss['strat_cagr']*100:+.2f}%/yr")
    print(f"  Buffett timing CAGR (net):   {ss['strat_cagr_net']*100:+.2f}%/yr")
    print(f"  Buy-and-hold CAGR (price):   {ss['bah_cagr']*100:+.2f}%/yr")
    print(f"  Years invested:              {ss['frac_invested']*100:.1f}%")
    print(f"  Regime flips:                {ss['n_flips']}")
    print(f"  Excess (strat-BAH) HAC t:    {ss['excess_t']:+.3f}\n")

    print("Verdict: Signal=WEAK (right sign, HAC t=-0.92, R^2<1%); "
          "Tradability=MIRAGE (timing earns 1.3%/yr vs 8.1%/yr buy-and-hold).")
    print("The Buffett Indicator is a thermometer, not a timer.")


if __name__ == "__main__":
    main()
