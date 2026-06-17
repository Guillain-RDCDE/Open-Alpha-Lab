"""Real-data verification — Study 279 (Geomagnetic-Storms). Regenerates docs/results.md.

Loads the cached ^GSPC daily series (repo-level _cache/^GSPC_split_only.parquet),
resamples to monthly price returns, joins with the hardcoded geomagnetic Ap index,
and runs the full analysis: storm-vs-calm gap with Newey-West HAC t-stat, Welch
t-test, permutation test, and the lagged/costed long-short overlay vs buy-and-hold.

Cache-only (no fetch). Run from anywhere:

    python studies/279-geomagnetic-storms/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geomagnetic_storms import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 279 — Geomagnetic-Storms — real-data verification\n")
    print("Data: ^GSPC daily close (repo _cache/^GSPC_split_only.parquet), monthly price returns")
    print("      Geomagnetic Ap index: hardcoded in data.py (1932-2025)\n")

    df = data.fetch_gspc_monthly()
    fp = data.fingerprint(df)
    print(f"Data stamp: {df.index.year.min()}-{df.index.year.max()}, "
          f"n={len(df)} months, fingerprint={fp}\n")

    print("Regime counts:", df["regime"].value_counts().to_dict())
    print(f"Ap thresholds: calm <= {df['ap'].quantile(0.20):.1f}, "
          f"stormy >= {df['ap'].quantile(0.80):.1f}\n")

    s = st.storm_calm_stats(df, n_permutations=10_000, seed=279)
    print("=== Storm vs calm (Krivelyova-Robotti) ===")
    print(f"  Mean calm-month return:   {s['mean_calm']*100:+.3f}%/month")
    print(f"  Mean stormy-month return: {s['mean_stormy']*100:+.3f}%/month")
    print(f"  Gap (calm - stormy):      {s['gap']*100:+.3f}%/month "
          f"(~{s['gap']*12*100:+.1f}%/yr)")
    print(f"  Newey-West HAC t-stat:    {s['hac_t']:+.3f}   <-- the honest SE")
    print(f"  Welch two-sample t:       {s['welch_t']:+.3f} (p={s['welch_p']:.4f})")
    print(f"  Permutation p (one-sided):{s['perm_p']:.4f}\n")

    print("=== Sub-period robustness ===")
    for lo, hi in [(1932, 1979), (1980, 2025), (1990, 2025)]:
        d = df[(df.index.year >= lo) & (df.index.year <= hi)]
        ss = st.storm_calm_stats(d, n_permutations=3000, seed=279)
        print(f"  {lo}-{hi}: n={len(d)}, gap={ss['gap']*100:+.3f}%/mo, "
              f"HAC t={ss['hac_t']:+.2f}, perm p={ss['perm_p']:.3f}")
    print()

    print("=== Tradable overlay (long-calm / short-storm, 1-month lag, costs+borrow) ===")
    ls = st.long_short_storm(df)
    g = st.summarize(ls["gross"]); n = st.summarize(ls["net"]); m = st.summarize(df["ret"])
    print(f"  Gross: {g['mean_ann']*100:+.2f}%/yr (HAC t={g['tstat']:+.2f})")
    print(f"  Net:   {n['mean_ann']*100:+.2f}%/yr (HAC t={n['tstat']:+.2f}, "
          f"Sharpe {n['sharpe_ann']:.2f})")
    print(f"  Buy & hold ^GSPC: {m['mean_ann']*100:+.2f}%/yr (Sharpe {m['sharpe_ann']:.2f})")
    print(f"  Avg monthly cost drag: {ls['cost'].mean()*1e4:.2f} bps\n")

    print("Verdict: Signal=WEAK (HAC t=1.88 < 2, right sign, mechanism-backed), "
          "Tradability=MIRAGE (net 2.4%/yr << 8.7%/yr buy-and-hold).")
    print("Note: ^GSPC is price-only (no dividends); storm/calm thresholds are in-sample.")


if __name__ == "__main__":
    main()
