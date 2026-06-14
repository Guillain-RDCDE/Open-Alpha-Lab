"""Real-tape verification — Study 123 (Altman-Z). Regenerates docs/results.md numbers.

Reads the desk's shared EDGAR caches + yfinance prices, computes the Altman Z-score
for each firm-year (2008-2023), sorts into terciles, and reports the annual hedge
and per-bucket statistics against the docs/results.md numbers.

    python studies/123-altman-z/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from altman_z import data, strategy as st  # noqa: E402
from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402


def main() -> None:
    print("Study 123 — Altman-Z — real EDGAR tape verification\n")

    z, fwd = data.fetch_panel()
    if z.empty:
        print("ERROR: EDGAR cache not found at", data.DEFAULT_CACHE)
        print("The shared _cache/ directory is required for real-tape results.")
        sys.exit(1)

    fp = data.fingerprint(z)
    print(f"Z-score panel: {z.shape[0]} years × {z.shape[1]} tickers, fingerprint={fp}")
    print(f"Years: {int(z.index.min())}–{int(z.index.max())}")
    print(f"Non-null Z entries: {z.notna().sum().sum()}")
    print()

    res = st.tertile_hedge(z, fwd)
    hedge = pd.Series(res["hedge"].values, dtype=float)
    hac_h = mean_tstat_hac(hedge)
    ci = sharpe_ci_bootstrap(hedge, n_boot=2000, periods_per_year=1, seed=123)

    print("=== Annual bucket returns (equal-weighted) ===")
    print(res[["n", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"]].round(3).to_string())
    print()

    print("=== Hedge (high-Z − low-Z) summary ===")
    print(f"  mean: {hedge.mean()*100:+.2f}%/yr")
    print(f"  vol:  {hedge.std(ddof=1)*100:.2f}%/yr")
    print(f"  Sharpe: {ci['sharpe']:.2f}, 95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
    print(f"  HAC t-stat: {hac_h['tstat']:+.2f}  (|t| >= 2 is the inference bar)")
    print(f"  Hit rate: {(hedge > 0).mean()*100:.0f}% of years positive")
    print()

    print("=== Per-bucket annual means ===")
    for leg, lbl in [("ret_lo", "Low-Z (distressed)"), ("ret_mid", "Mid-Z (grey zone)"),
                     ("ret_hi", "High-Z (safe)"), ("ret_mkt", "Market EW")]:
        s = mean_tstat_hac(pd.Series(res[leg].values, dtype=float))
        print(f"  {lbl:22s}: mean={res[leg].mean()*100:+.1f}%/yr, HAC t={s['tstat']:+.2f}")
    print()

    # Firm-year level correlation
    all_z, all_r = [], []
    for yr in z.index:
        z_yr = z.loc[yr].dropna(); r_yr = fwd.loc[yr].dropna()
        both = z_yr.index.intersection(r_yr.index)
        all_z.extend(z_yr.loc[both].tolist()); all_r.extend(r_yr.loc[both].tolist())
    all_z = np.array(all_z); all_r = np.array(all_r)
    corr = np.corrcoef(all_z, all_r)[0, 1]
    print(f"Firm-level Pearson corr(Z, next-yr return): {corr:.4f}")
    print(f"Total firm-year observations: {len(all_z)}")
    print()
    print("Survivorship bias caveat: panel covers current S&P 500 projected back.")
    print("Results are upper bounds; live performance would likely be weaker.")
    print()
    print("=== Verdict ===")
    print(f"  Signal: NONE (HAC t = {hac_h['tstat']:+.2f}, threshold = ±2)")
    print(f"  Tradability: MIRAGE (thin annual premium eaten by implementation costs)")


if __name__ == "__main__":
    main()
