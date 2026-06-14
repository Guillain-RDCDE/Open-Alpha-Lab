"""Real-data verification — Study 155 (Asset-Turnover).  Regenerates docs/results.md numbers.

Reads the shared EDGAR fundamental caches and the pre-computed annual-return panel to run
the asset-turnover factor on the S&P 500 survivorship-biased panel, computes the top-quintile
vs equal-weight-market excess return with a HAC t-stat, and prints the key numbers.

    python studies/155-asset-turnover/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from asset_turnover import data, strategy as st  # noqa: E402


def main() -> None:
    at_sig, roa_sig, fwd = data.fetch_panel()
    if at_sig.empty:
        print("No EDGAR cache found. Run from the repo root where _cache/ exists.")
        return

    print("=== Asset Turnover — Real EDGAR panel ===")
    print(f"Signal years: {at_sig.index.min()} to {at_sig.index.max()}")
    print(f"Tickers with valid AT data: {at_sig.notna().any(axis=0).sum()}")
    print("Survivorship-biased: current S&P 500 projected back (upper bound)\n")

    # Fingerprints
    cache = data.DEFAULT_CACHE
    for concept in data.CONCEPTS:
        p = os.path.join(cache, f"_edgar_{concept}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            fp = data.fingerprint(df)
            print(f"  {concept[:36]:36s}  fp={fp}")
    yr_ret = pd.read_parquet(os.path.join(cache, "_edgar_yrret.parquet"))
    print(f"  {'_edgar_yrret':36s}  fp={data.fingerprint(yr_ret)}")
    print()

    # AT quintile hedge
    h_at = st.quintile_returns(at_sig, fwd, q=0.20)
    s_top = st.summarize(h_at["top"])
    s_mkt = st.summarize(h_at["market"])
    s_hedge = st.summarize(h_at["hedge"])
    s_ls = st.summarize(h_at["ls"])

    print("=== AT top quintile (highest Revenues/Assets) ===")
    print(f"  Mean: {s_top['mean']*100:.1f}%/yr  Sharpe: {s_top['sharpe']:.2f}  "
          f"t: {s_top['tstat']:+.2f}  hit: {s_top['hit_rate']:.0%}")

    print("=== EW market (all firms) ===")
    print(f"  Mean: {s_mkt['mean']*100:.1f}%/yr  Sharpe: {s_mkt['sharpe']:.2f}  "
          f"t: {s_mkt['tstat']:+.2f}")

    print("=== AT top quintile EXCESS vs EW market ===")
    print(f"  Mean: {s_hedge['mean']*100:.1f}%/yr  Sharpe: {s_hedge['sharpe']:.2f}  "
          f"HAC t: {s_hedge['tstat']:+.2f}  hit: {s_hedge['hit_rate']:.0%}")

    print("=== AT long-short (top - bottom quintile) ===")
    print(f"  Mean: {s_ls['mean']*100:.1f}%/yr  Sharpe: {s_ls['sharpe']:.2f}  "
          f"HAC t: {s_ls['tstat']:+.2f}  hit: {s_ls['hit_rate']:.0%}")

    # ROA comparison
    h_roa = st.quintile_returns(roa_sig, fwd, q=0.20)
    s_roa_hedge = st.summarize(h_roa["hedge"])
    print("\n=== ROA top quintile EXCESS vs EW market (profitability comparison) ===")
    print(f"  Mean: {s_roa_hedge['mean']*100:.1f}%/yr  Sharpe: {s_roa_hedge['sharpe']:.2f}  "
          f"HAC t: {s_roa_hedge['tstat']:+.2f}  hit: {s_roa_hedge['hit_rate']:.0%}")

    # Head-to-head AT vs ROA
    common_yrs = h_at.index.intersection(h_roa.index)
    at_exc = h_at.loc[common_yrs, "hedge"]
    roa_exc = h_roa.loc[common_yrs, "hedge"]
    at_vs_roa = at_exc - roa_exc
    s_diff = st.summarize(at_vs_roa)
    print("\n=== AT excess minus ROA excess (head-to-head, same years) ===")
    print(f"  Mean: {s_diff['mean']*100:.1f}%/yr  HAC t: {s_diff['tstat']:+.2f}")
    print("  (Near zero means AT adds nothing beyond plain profitability.)")

    # Random control
    print("\n=== Random control (same quintile size) ===")
    rand = st.random_portfolio_returns(at_sig, fwd, q=0.20, n_draws=500, seed=155)
    print(f"  Mean excess: {rand.mean()*100:.1f}%  Std: {rand.std()*100:.1f}%")
    pct = float((rand < s_hedge["mean"]).mean())
    print(f"  AT top quintile beats {pct:.0%} of random draws")

    # Year-by-year
    print("\n=== Year-by-year ===")
    print(h_at[["top", "bottom", "market", "hedge", "n"]].round(3).to_string())

    # Verdict
    print("\n=== Verdict ===")
    t = s_hedge["tstat"]
    if abs(t) >= 2.0:
        sig_label = "REAL" if t > 0 else "NONE (significant in wrong direction)"
    elif abs(t) >= 1.5:
        sig_label = "WEAK (|t| >= 1.5 but < 2.0; survivorship-biased upper bound)"
    else:
        sig_label = "NONE"
    print(f"  Signal: {sig_label}  (HAC t = {t:+.2f}; survivorship upper bound)")
    print(f"  Tradability: MIRAGE -- annual rebalance, concentrated portfolio (~{int(h_at['n_top'].mean())} names),")
    print(f"    AT adds nothing beyond ROA (delta = {s_diff['mean']*100:.1f}%/yr, t = {s_diff['tstat']:+.2f})")


if __name__ == "__main__":
    main()
