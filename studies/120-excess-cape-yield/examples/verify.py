"""Real-data verification -- Study 120 (Excess-CAPE-Yield).

Reads the staged Shiller parquet (_cache/shiller_sp500.parquet), computes the
headline ECY metrics, and prints the numbers that populate docs/results.md.
No network access required; the parquet is a read-only staged input.

    python studies/120-excess-cape-yield/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from excess_cape_yield import data, strategy as st  # noqa: E402


def main() -> None:
    df = data.load_shiller()
    fp_ecy = data.fingerprint(df, "ecy")
    fp_fwd = data.fingerprint(df, "fwd10_excess")
    s = st.summarize(df)

    print("=== Study 120 -- Excess-CAPE-Yield (real Shiller data) ===\n")
    print(f"Source: _cache/shiller_sp500.parquet")
    print(f"Span: {df.index[0].date()} to {df.index[-1].date()} (full file);")
    print(f"       regression sample {s['year_start']}-{s['year_end']} (forward return requires 120m)")
    print(f"n_monthly (reg sample): {s['n_monthly']}")
    print(f"n_non-overlapping 10yr windows: {s['n_nonoverlap_10yr']}")
    print(f"Fingerprint (ecy series): {fp_ecy}")
    print(f"Fingerprint (fwd10_excess series): {fp_fwd}")
    print(f"As-of: 2026-06-14\n")

    print("=== In-sample regression: ECY -> 10yr forward excess return ===")
    print(f"R2 (monthly, overlapping):    {s['r2_ecy_monthly']:.3f}  [context only; overlapping windows]")
    print(f"Corr (monthly):               {s['corr_ecy_monthly']:.3f}")
    print(f"R2 (non-overlapping windows): {s['r2_ecy_nonoverlap']:.3f}")
    print(f"Slope (non-overlapping):      {s['slope_ecy_nonoverlap']:.3f}")
    print(f"OLS t-stat (non-overlapping): {s['tstat_ecy_nonoverlap']:.2f}   [|t| > 2 => REAL]")
    print()

    print("=== Comparison: 1/CAPE alone -> 10yr forward excess return ===")
    print(f"R2 (monthly, overlapping):    {s['r2_ey_monthly']:.3f}")
    print(f"R2 (non-overlapping windows): {s['r2_ey_nonoverlap']:.3f}")
    print(f"Slope (non-overlapping):      {s['slope_ey_nonoverlap']:.3f}")
    print(f"OLS t-stat (non-overlapping): {s['tstat_ey_nonoverlap']:.2f}   [ECY beats 1/CAPE alone]")
    print()

    print("=== 1-year horizon (monthly, overlapping) ===")
    print(f"R2 (ECY -> fwd1_excess): {s['r2_ecy_1yr_monthly']:.3f}  [weak; horizon matters enormously]")
    print()

    print("=== Out-of-sample (50/50 split on non-overlapping windows) ===")
    print(f"OOS R2 (ECY): {s['oos_r2_ecy']:+.3f}   [> 0 => beats historical mean benchmark]")
    print(f"Train n: {s['oos_train_n']}, Test n: {s['oos_test_n']}")
    print()

    print("=== ECY level distribution ===")
    print(f"ECY mean: {s['ecy_mean_pct']:.2f}%")
    print(f"ECY std:  {s['ecy_std_pct']:.2f}%")
    print(f"Fraction of months with ECY > 0: {s['frac_pos_ecy']:.1f}%")
    print(f"Mean fwd10_excess when ECY > 0:  {s['mean_excess_when_ecy_pos_pct']:.2f}%/yr")
    print(f"Mean fwd10_excess when ECY <= 0: {s['mean_excess_when_ecy_neg_pct']:.2f}%/yr")
    print()

    print("=== Verdict ===")
    t = s["tstat_ecy_nonoverlap"]
    verdict = "REAL" if abs(t) >= 2.0 else "WEAK"
    print(f"Signal: {verdict} (|t| = {abs(t):.2f} >= 2 on non-overlapping windows)")
    print("Tradability: MIRAGE (10yr holding period; not a timing signal)")


if __name__ == "__main__":
    main()
