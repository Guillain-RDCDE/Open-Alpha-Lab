"""Real-tape verification — Study 229 (Beneish M-score). Regenerates docs/results.md numbers.

Reads the desk's shared EDGAR caches + study-local EDGAR caches (fetched on first run),
computes the Beneish M-score for each firm-year (2008–2023), sorts into terciles
(long low-M / short high-M), and reports the annual hedge and per-bucket statistics.

    python studies/229-beneish-m-score/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from beneish_m_score import data, strategy as st  # noqa: E402
from quantlab.analytics import mean_tstat_hac  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402


def main() -> None:
    print("Study 229 — Beneish M-score — real EDGAR tape verification\n")

    m, fwd = data.fetch_panel()
    if m.empty:
        print("ERROR: EDGAR caches not found.")
        print("  Shared cache:", data.SHARED_CACHE)
        print("  Local cache: ", data.LOCAL_CACHE)
        print("Run this script first with internet access to auto-fetch AR and PPE data.")
        sys.exit(1)

    fp = data.fingerprint(m)
    n_valid = int(m.notna().sum().sum())
    print(f"M-score panel: {m.shape[0]} years × {m.shape[1]} tickers, fingerprint={fp}")
    print(f"Years: {int(m.index.min())}–{int(m.index.max())}")
    print(f"Non-null M entries: {n_valid}")
    print()

    # M-score distribution
    mvals = m.values.flatten().astype(float)
    mvals = mvals[np.isfinite(mvals)]
    pct_manip = (mvals > data.M_THRESHOLD).mean() * 100
    pct_clean  = (mvals <= data.M_THRESHOLD).mean() * 100
    print(f"M > -1.78 (likely manipulators): {pct_manip:.1f}% of firm-years")
    print(f"M <= -1.78 (likely clean):        {pct_clean:.1f}% of firm-years")
    print(f"Mean M-score: {mvals.mean():.2f}, Median: {np.median(mvals):.2f}")
    print()

    res = st.tertile_hedge(m, fwd)
    hedge = pd.Series(res["hedge"].values, dtype=float)
    hac_h = mean_tstat_hac(hedge)
    ci = sharpe_ci_bootstrap(hedge, n_boot=2000, periods_per_year=1, seed=229)

    print("=== Annual bucket returns (equal-weighted) ===")
    print(res[["n", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"]].round(3).to_string())
    print()

    print("=== Hedge (low-M minus high-M) summary ===")
    print(f"  mean:   {hedge.mean()*100:+.2f}%/yr")
    print(f"  vol:    {hedge.std(ddof=1)*100:.2f}%/yr")
    print(f"  Sharpe: {ci['sharpe']:.2f}, 95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
    print(f"  HAC t-stat: {hac_h['tstat']:+.2f}  (|t| >= 2 is the inference bar)")
    print(f"  Hit rate: {(hedge > 0).mean()*100:.0f}% of years positive")
    print()

    print("=== Per-bucket annual means ===")
    for leg, lbl in [
        ("ret_lo",  "Low-M  (clean)   "),
        ("ret_mid", "Mid-M  (grey)    "),
        ("ret_hi",  "High-M (manipulators)"),
        ("ret_mkt", "Market EW        "),
    ]:
        s = mean_tstat_hac(pd.Series(res[leg].values, dtype=float))
        print(f"  {lbl}: mean={res[leg].mean()*100:+.1f}%/yr, HAC t={s['tstat']:+.2f}")
    print()

    # Firm-level correlation
    all_m, all_r = [], []
    for yr in m.index:
        m_yr = m.loc[yr].dropna()
        r_yr = fwd.loc[yr].dropna()
        both = m_yr.index.intersection(r_yr.index)
        all_m.extend(m_yr.loc[both].tolist())
        all_r.extend(r_yr.loc[both].tolist())
    all_m = np.array(all_m)
    all_r = np.array(all_r)
    corr = np.corrcoef(all_m, all_r)[0, 1]
    print(f"Firm-level Pearson corr(M, next-yr return): {corr:.4f}")
    print(f"Total firm-year observations: {len(all_m)}")
    print()
    print("Fingerprint:", fp)
    print()
    print("=== Verdict ===")
    t = hac_h['tstat']
    if abs(t) >= 2.0:
        print(f"  Signal: WEAK (HAC t = {t:+.2f}, reaches bar but on survivorship-biased panel)")
    else:
        print(f"  Signal: NONE (HAC t = {t:+.2f}, below ±2 inference bar)")
    print("  Tradability: MIRAGE (survivorship bias, thin spread, short-selling costs)")


if __name__ == "__main__":
    main()
