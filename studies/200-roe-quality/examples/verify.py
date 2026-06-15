"""Real-tape verification — Study 200 (ROE-Quality). Regenerates docs/results.md numbers.

Reads the EDGAR cache, computes the ROE and GP/Assets signals, runs quintile sorts and
the random-portfolio control, and prints the headline statistics. No network access
required — reads from the shared ``_cache/`` parquet files.

    python studies/200-roe-quality/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roe_quality import data, strategy as st  # noqa: E402


def main() -> None:
    roe, gpr, fwd = data.fetch_panel()
    if roe.empty:
        print("ERROR: EDGAR cache not found. Place _edgar_*.parquet files in _cache/.")
        sys.exit(1)

    print(f"ROE panel: {len(roe.columns)} tickers, {len(roe)} years "
          f"({roe.index.min()}–{roe.index.max()})")
    print(f"Fingerprint (ROE): {data.fingerprint(roe)}")
    print(f"Fingerprint (GP):  {data.fingerprint(gpr)}")
    print()

    # Restrict to 2009-2025 for a clean 17-year window (enough EDGAR coverage)
    roe_w = roe.loc[2009:]
    gpr_w = gpr.loc[2009:]
    fwd_w = fwd.loc[2009:]

    h_roe = st.quintile_returns(roe_w, fwd_w, q=0.20)
    h_gp = st.quintile_returns(gpr_w, fwd_w, q=0.20)

    print("=== ROE quintile returns (low -> high, forward 1yr) ===")
    for q in ["q1", "q2", "q3", "q4", "q5", "market"]:
        s = st.summary(h_roe[q])
        label = "low-ROE" if q == "q1" else ("high-ROE" if q == "q5" else
                ("EW mkt" if q == "market" else q))
        print(f"  {label:10s} mean={s['mean']*100:+.1f}%/yr  Sharpe={s['sharpe']:+.2f}"
              f"  HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}")

    hedge_roe = st.summary(h_roe["hedge"])
    print()
    print("=== long high-ROE / short low-ROE hedge (Q5 - Q1) ===")
    print(f"  mean={hedge_roe['mean']*100:+.1f}%/yr  Sharpe={hedge_roe['sharpe']:+.2f}  "
          f"HAC t={hedge_roe['tstat']:+.2f}  hit={hedge_roe['hit_rate']:.0%}  n={hedge_roe['n']}")

    hi_excess_s = st.summary(h_roe["q5"] - h_roe["market"])
    lo_excess_s = st.summary(h_roe["q1"] - h_roe["market"])
    print()
    print("=== excess returns vs EW market ===")
    print(f"  High-ROE (q5) excess: mean={hi_excess_s['mean']*100:+.1f}%/yr  "
          f"HAC t={hi_excess_s['tstat']:+.2f}")
    print(f"  Low-ROE (q1) excess:  mean={lo_excess_s['mean']*100:+.1f}%/yr  "
          f"HAC t={lo_excess_s['tstat']:+.2f}")

    print()
    print("=== random-portfolio null distribution ===")
    rand = st.random_portfolio_returns(roe_w, fwd_w, q=0.20, n_draws=500, seed=200)
    actual_hi_excess = float((h_roe["q5"] - h_roe["market"]).mean())
    pct_beaten = float((rand > actual_hi_excess).mean())
    print(f"  Random control: mean={rand.mean()*100:+.2f}%  std={rand.std()*100:.1f}%")
    print(f"  High-ROE portfolio beats {pct_beaten:.0%} of random same-size draws")

    print()
    print("=== GP/Assets (gross profitability) head-to-head ===")
    hedge_gp = st.summary(h_gp["hedge"])
    gp_hi_exc = st.summary(h_gp["q5"] - h_gp["market"])
    print(f"  GP hedge (Q5-Q1): mean={hedge_gp['mean']*100:+.1f}%/yr  "
          f"HAC t={hedge_gp['tstat']:+.2f}")
    print(f"  GP Q5 excess: mean={gp_hi_exc['mean']*100:+.1f}%/yr  "
          f"HAC t={gp_hi_exc['tstat']:+.2f}")

    print()
    print("=== year-by-year summary ===")
    print(f"{'year':>6}  {'n':>4}  {'q1 (lo)':>9}  {'q5 (hi)':>9}  "
          f"{'mkt':>9}  {'hedge':>9}")
    for y, row in h_roe.iterrows():
        print(f"{y:>6}  {row['n']:>4.0f}  {row['q1']*100:>+8.1f}%"
              f"  {row['q5']*100:>+8.1f}%  {row['market']*100:>+8.1f}%"
              f"  {row['hedge']*100:>+8.1f}%")

    print()
    print("SURVIVORSHIP BIAS WARNING: panel covers CURRENT S&P 500 members only.")
    print("All results are UPPER BOUNDS in magnitude on a live strategy.")


if __name__ == "__main__":
    main()
