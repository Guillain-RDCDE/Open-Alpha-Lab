"""Real-data verification — Study 121 (Magic-Formula). Regenerates docs/results.md numbers.

Reads the shared EDGAR fundamental caches and the pre-computed annual-return panel to run
Greenblatt's Magic Formula on the S&P 500 survivorship-biased panel, computes the top-decile
vs equal-weight-market excess return with a HAC t-stat, and prints the key numbers.

    python studies/121-magic-formula/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from magic_formula import data, strategy as st  # noqa: E402


def main() -> None:
    sig, fwd = data.fetch_panel()
    if sig.empty:
        print("No EDGAR cache found. Run from the repo root where _cache/ exists.")
        return

    print("=== Magic Formula — Real EDGAR panel ===")
    print(f"Signal years: {sig.index.min()} – {sig.index.max()}")
    print(f"Tickers with full data: {sig.notna().any(axis=0).sum()}")
    print(f"Survivorship-biased: current S&P 500 projected back (upper bound)\n")

    # Fingerprints
    cache = data.DEFAULT_CACHE
    for concept in data.CONCEPTS:
        p = os.path.join(cache, f"_edgar_{concept}.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p)
            fp = data.fingerprint(df)
            print(f"  {concept[:30]:30s}  fp={fp}")
    yr_ret = pd.read_parquet(os.path.join(cache, "_edgar_yrret.parquet"))
    print(f"  {'_edgar_yrret':30s}  fp={data.fingerprint(yr_ret)}")
    print()

    # Top-decile vs market
    h = st.quantile_hedge(sig, fwd, q=0.10)
    s_top = st.summary(h["top"])
    s_mkt = st.summary(h["market"])
    s_hedge = st.summary(h["hedge"])
    s_ls = st.summary(h["top"] - h["bottom"])

    print("=== Top decile (best MF rank) ===")
    print(f"  Mean: {s_top['mean']*100:.1f}%/yr  Sharpe: {s_top['sharpe']:.2f}  t: {s_top['tstat']:+.2f}  hit: {s_top['hit_rate']:.0%}")

    print("=== EW market (all firms) ===")
    print(f"  Mean: {s_mkt['mean']*100:.1f}%/yr  Sharpe: {s_mkt['sharpe']:.2f}  t: {s_mkt['tstat']:+.2f}")

    print("=== Top decile EXCESS vs EW market ===")
    print(f"  Mean: {s_hedge['mean']*100:.1f}%/yr  Sharpe: {s_hedge['sharpe']:.2f}  HAC t: {s_hedge['tstat']:+.2f}  hit: {s_hedge['hit_rate']:.0%}")

    print("=== Long-short (top - bottom decile) ===")
    print(f"  Mean: {s_ls['mean']*100:.1f}%/yr  Sharpe: {s_ls['sharpe']:.2f}  HAC t: {s_ls['tstat']:+.2f}  hit: {s_ls['hit_rate']:.0%}")

    print("\n=== Random control (same decile size) ===")
    rand = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=500, seed=121)
    print(f"  Mean excess: {rand.mean()*100:.1f}%  Std: {rand.std()*100:.1f}%")
    pct = float((rand < s_hedge["mean"]).mean())
    print(f"  Top decile beats {pct:.0%} of random draws")

    print("\n=== Year-by-year ===")
    print(h[["top", "bottom", "market", "hedge", "n"]].round(3).to_string())

    print(f"\n=== Verdict ===")
    t = s_hedge["tstat"]
    if abs(t) >= 2.0:
        sig_label = "REAL" if t > 0 else "NONE (significant in the wrong direction)"
    else:
        sig_label = "NONE" if abs(t) < 1.0 else "WEAK"
    print(f"  Signal: {sig_label}  (HAC t = {t:+.2f}; survivorship upper bound)")
    print(f"  Tradability: FRAGILE at best — annual rebalance is feasible in theory,")
    print(f"    but the S&P 500 survivor panel overstates any edge, and the small")
    print(f"    portfolio (~16 names) has high idiosyncratic risk (sharpe={s_top['sharpe']:.2f} raw).")


if __name__ == "__main__":
    main()
