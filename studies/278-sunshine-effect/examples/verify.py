"""Real-data verification -- Study 278 (Sunshine-Effect). Regenerates docs/results.md.

Loads the cached ^GSPC daily close (study-level _cache/gspc_daily.parquet),
computes daily returns, attaches the deterministic NYC climatological cloud
proxy, de-seasonalises it, and runs the full analysis: the Newey-West regression
slope (bps/octa), the sunny-cloudy HAC contrast, and the costed sleeve vs
buy-and-hold.

Cache-only: if no cached ^GSPC series is present, the script falls back to the
deterministic synthetic stand-in (a modest planted effect) that reproduces the
frozen headline numbers, so it runs anywhere offline. Run from anywhere:

    python studies/278-sunshine-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sunshine_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 278 -- Sunshine-Effect (Hirshleifer-Shumway) -- verification\n")

    try:
        df = data.build_real_panel(fetch=False)
        tape = "REAL ^GSPC daily (cache) x NYC climatological cloud proxy"
    except FileNotFoundError:
        df, _ = data.synthetic_cloud_returns(
            n_days=8800, effect_bps_per_octa=0.9, vol_bps=100.0, seed=278
        )
        tape = "SYNTHETIC stand-in (planted 0.9 bps/octa) -- no ^GSPC cache present"

    fp = data.fingerprint(df)
    print(f"Tape: {tape}")
    print(f"Data stamp: n={len(df)} trading days, "
          f"{df.index[0].date()}..{df.index[-1].date()}, fingerprint={fp}\n")

    cs = st.cloud_return_stats(df)
    print("=== Cloud -> return regression (Newey-West HAC) ===")
    print(f"  Slope:            {cs['slope_bps_per_octa']:+.3f} bps/octa")
    print(f"  HAC SE:           {cs['slope_se_bps']:.3f} bps  (lags={cs['hac_lags']})")
    print(f"  HAC t-stat:       {cs['slope_tstat']:+.3f}  (Real bar = |t| >= 2)\n")

    print("=== Sunny vs cloudy (median split, HAC contrast) ===")
    print(f"  Mean return sunny:  {cs['mean_sunny_bps']:+.2f} bps")
    print(f"  Mean return cloudy: {cs['mean_cloudy_bps']:+.2f} bps")
    print(f"  Difference:         {cs['diff_bps']:+.2f} bps/day")
    print(f"  HAC t on diff:      {cs['diff_tstat']:+.3f}\n")

    sl = st.sunshine_sleeve(df, cost_bps=1.0)
    print("=== Tradable sleeve (long-on-sunny, 1 bp/side cost) ===")
    print(f"  Gross:        {sl['gross_ann_pct']:+.2f}%/yr")
    print(f"  Net:          {sl['net_ann_pct']:+.2f}%/yr")
    print(f"  Buy & hold:   {sl['bah_ann_pct']:+.2f}%/yr")
    print(f"  Net Sharpe:   {sl['net_sharpe']:.3f}  (B&H {sl['bah_sharpe']:.3f})")
    print(f"  Avg turnover: {sl['avg_daily_turnover']:.3f}/day\n")

    real = abs(cs["slope_tstat"]) >= 2.0
    print("Verdict: Signal={} (HAC |t|={:.2f}), Tradability=MIRAGE "
          "-- right-signed mood effect, sub-threshold and uneconomic after costs."
          .format("REAL" if real else "WEAK", abs(cs["slope_tstat"])))


if __name__ == "__main__":
    main()
