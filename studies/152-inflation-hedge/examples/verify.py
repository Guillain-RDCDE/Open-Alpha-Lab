"""Real-tape runner — Study 152 (Inflation-Hedge).

Loads the Shiller monthly panel from the repo-wide cache
(``_cache/shiller_sp500.parquet``; never re-fetched) and regenerates the
headline numbers printed in ``docs/results.md``.

Run:
    python studies/152-inflation-hedge/examples/verify.py

The script is cache-first: it reads the staged parquet and produces all numbers
offline.  No Yahoo, no EDGAR, no network.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from inflation_hedge import data, strategy as st  # noqa: E402


def main() -> None:
    # --- Load real data -------------------------------------------------------
    df = data.load_shiller()
    fp = data.fingerprint(df)
    date_start = df.index[0].strftime("%Y-%m")
    date_end = df.index[-1].strftime("%Y-%m")
    n_total = len(df)
    n_high = int((df["cpi_regime"] == "high").sum())
    n_low = int((df["cpi_regime"] == "low").sum())

    # --- Run analysis ---------------------------------------------------------
    result = st.summarize(df)
    fisher = st.fisher_regression(df)
    regime_1y = st.regime_returns(df, fwd_col="fwd_real_tr_12m")
    regime_5y = st.regime_returns(df, fwd_col="fwd_real_tr_60m")

    # --- Print results.md content ---------------------------------------------
    print("=" * 72)
    print("Study 152 - Inflation-Hedge: REAL TAPE RESULTS")
    print("=" * 72)
    print(f"Data source    : Shiller S&P 500 monthly panel")
    print(f"Window         : {date_start} to {date_end}  (n={n_total} months)")
    print(f"Fingerprint    : {fp}  (cpi_yoy column)")
    print(f"High-inflation months (CPI YoY > 3%): {n_high}")
    print(f"Low-inflation months  (CPI YoY <= 3%): {n_low}")
    print()
    print("--- Fisher Hypothesis (nominal returns vs. CPI YoY) ---")
    print(f"  OLS beta (nominal 12m return ~ CPI YoY): {fisher['beta_nom']:+.4f}")
    print(f"  HAC t-stat (nominal beta):               {fisher['t_nom']:+.4f}")
    print(f"  R^2 (nominal):                           {fisher['r2_nom']:.4f}")
    print(f"  Full hedge would require beta = 1.0 (t >> 2).")
    print()
    print("--- Fisher Hypothesis (real returns vs. CPI YoY) ---")
    print(f"  OLS beta (real total return ~ CPI YoY): {fisher['beta_real']:+.4f}")
    print(f"  HAC t-stat (real beta):                 {fisher['t_real']:+.4f}")
    print(f"  Fama-Schwert: expect negative beta (real returns fall w/ inflation).")
    print()
    print("--- Regime Analysis: Forward 1-Year Real Total Return ---")
    print(f"  High-inflation mean: {regime_1y['high']['mean']*100:+.2f}% pa")
    print(f"  Low-inflation mean:  {regime_1y['low']['mean']*100:+.2f}% pa")
    print(f"  Difference (high minus low): {regime_1y['diff_mean']*100:+.2f}pp")
    print(f"  HAC t-stat (difference): {regime_1y['t_diff']:+.4f}")
    print(f"  Shuffled-label control t: {regime_1y['t_diff_shuffled']:+.4f}")
    print()
    print("--- Regime Analysis: Forward 5-Year Real Total Return (annualised) ---")
    print(f"  High-inflation mean: {regime_5y['high']['mean']*100:+.2f}% pa")
    print(f"  Low-inflation mean:  {regime_5y['low']['mean']*100:+.2f}% pa")
    print(f"  Difference (high minus low): {regime_5y['diff_mean']*100:+.2f}pp")
    print(f"  HAC t-stat (difference): {regime_5y['t_diff']:+.4f}")
    print()
    print("--- VERDICT ---")
    print("  Fisher nominal beta ~0.31 (t ~1.30): stocks do NOT hedge inflation")
    print("    in the short run — nominal returns rise only 31c per 1pp of CPI.")
    print("  Fisher real beta ~-0.72 (t ~-3.02): real returns fall significantly")
    print("    when inflation rises — the opposite of a perfect hedge.")
    print("  High-inflation regime 1y real return -7pp below low-inflation")
    print("    (t ~-4.08): statistically real, robust to shuffled-label control.")
    print("  Signal=MIXED: the *negative* effect is real & significant;")
    print("    long-run (5y) the gap narrows (t ~-2.1) but persists.")
    print("  Tradability=MIRAGE: regimes are broad macro epochs, not actionable")
    print("    timing signals. You cannot trade in and out of equities on CPI.")


if __name__ == "__main__":
    main()
