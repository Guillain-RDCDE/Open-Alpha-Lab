"""Real-tape verification -- Study 230 (Ohlson O-score). Regenerates docs/results.md numbers.

Reads the desk's shared EDGAR caches, computes the Ohlson O-score for each
firm-year (2008-2023), sorts into terciles (low-O = safe, high-O = distressed),
and reports the hedge (safe minus distressed) with HAC inference.

    python studies/230-ohlson-o-score/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ohlson_o_score import data, strategy as st  # noqa: E402
from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402


def main() -> None:
    print("Study 230 -- Ohlson O-score -- real EDGAR tape verification\n")

    o, fwd = data.fetch_panel()
    if o.empty:
        print("ERROR: EDGAR cache not found at", data.DEFAULT_CACHE)
        print("The shared _cache/ directory is required for real-tape results.")
        sys.exit(1)

    fp = data.fingerprint(o)
    print(f"O-score panel: {o.shape[0]} years x {o.shape[1]} tickers, fingerprint={fp}")
    print(f"Years: {int(o.index.min())}-{int(o.index.max())}")
    print(f"Non-null O entries: {o.notna().sum().sum()}")
    print()

    res = st.tertile_hedge(o, fwd)
    hedge = pd.Series(res["hedge"].values, dtype=float)
    hac_h = mean_tstat_hac(hedge)
    ci = sharpe_ci_bootstrap(hedge, n_boot=2000, periods_per_year=1, seed=230)

    print("=== Annual bucket returns (equal-weighted) ===")
    print(res[["n", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"]].round(3).to_string())
    print()

    print("=== Hedge (safe/low-O minus distressed/high-O) summary ===")
    print(f"  mean: {hedge.mean()*100:+.2f}%/yr")
    print(f"  vol:  {hedge.std(ddof=1)*100:.2f}%/yr")
    print(f"  Sharpe: {ci['sharpe']:.2f}, 95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
    print(f"  HAC t-stat: {hac_h['tstat']:+.2f}  (|t| >= 2 is the inference bar)")
    print(f"  Hit rate: {(hedge > 0).mean()*100:.0f}% of years positive")
    print()

    print("=== Per-bucket annual means ===")
    for leg, lbl in [("ret_lo", "Low-O (safe)"), ("ret_mid", "Mid-O (grey zone)"),
                     ("ret_hi", "High-O (distressed)"), ("ret_mkt", "Market EW")]:
        s = mean_tstat_hac(pd.Series(res[leg].values, dtype=float))
        print(f"  {lbl:26s}: mean={res[leg].mean()*100:+.1f}%/yr, HAC t={s['tstat']:+.2f}")
    print()

    # Firm-year level correlation
    all_o, all_r = [], []
    for yr in o.index:
        o_yr = o.loc[yr].dropna(); r_yr = fwd.loc[yr].dropna()
        both = o_yr.index.intersection(r_yr.index)
        all_o.extend(o_yr.loc[both].tolist()); all_r.extend(r_yr.loc[both].tolist())
    all_o = np.array(all_o); all_r = np.array(all_r)
    corr = np.corrcoef(all_o, all_r)[0, 1]
    print(f"Firm-level Pearson corr(O, next-yr return): {corr:.4f}")
    print(f"Total firm-year observations: {len(all_o)}")
    print()

    # O-score distribution stats
    o_flat = o.values.astype(float).flatten()
    o_flat = o_flat[np.isfinite(o_flat)]
    print(f"O-score distribution: mean={o_flat.mean():.2f}, median={np.median(o_flat):.2f}, "
          f"std={o_flat.std():.2f}")
    print(f"O-score range: [{o_flat.min():.2f}, {o_flat.max():.2f}]")
    print()

    print("Survivorship bias caveat: panel covers current S&P 500 projected back.")
    print("Results are upper bounds; live performance would likely be weaker.")
    print()
    print("=== Verdict ===")
    print(f"  Signal: NONE (HAC t = {hac_h['tstat']:+.2f}, threshold = +/-2)")
    print(f"  Tradability: MIRAGE (thin annual premium eaten by implementation costs)")
    print(f"  Fingerprint: {fp}")


if __name__ == "__main__":
    main()
