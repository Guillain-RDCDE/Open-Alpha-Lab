"""Real-data verification — Study 285 (St-Patricks-Day). Regenerates docs/results.md.

Loads ^GSPC daily prices from the shared repo cache (cache-only, no fetch),
derives the St. Patrick's Day trading sessions by date arithmetic, and runs the
full event study: HAC t-stat, Welch contrasts, placebo, Bonferroni day-of-March
sweep, sub-period split, and the cost-aware strategy P&L.

The ^GSPC parquet is at the repo-level _cache/^GSPC_split_only.parquet and is
read-only (no fetch required — it ships with the repo). Run from anywhere:

    python studies/285-st-patricks-day/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from st_patricks_day import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 285 — St-Patricks-Day bump — real-data verification\n")
    print("Data: ^GSPC daily close (price-only log-returns) from")
    print("      _cache/^GSPC_split_only.parquet (cache-only)")
    print("      St. Patrick sessions derived by date arithmetic in data.py\n")

    df = data.fetch_gspc(fetch=False)
    fp = data.fingerprint(df)
    print(f"Data stamp: {df.index.min().date()} -> {df.index.max().date()}, "
          f"n_days={len(df)}, fingerprint={fp}\n")

    s = st.summarize(df)

    print("=== Primary: St. Patrick session vs baselines ===")
    print(f"  n events:          {s['n_stpat']}")
    print(f"  Mean:              {s['mean_stpat_bps']:+.1f} bps   HAC t = {s['tstat_stpat_hac']:+.2f}")
    print(f"  vs all other days: contrast {s['contrast_vs_all_bps']:+.1f} bps  "
          f"Welch t = {s['tstat_welch_all']:+.2f}  p = {s['pval_welch_all']:.3f}")
    print(f"  vs other March:    contrast {s['contrast_vs_march_bps']:+.1f} bps  "
          f"Welch t = {s['tstat_welch_march']:+.2f}  p = {s['pval_welch_march']:.3f}\n")

    print("=== Placebo (March 24, one week later) ===")
    print(f"  n: {s['n_placebo']}  mean: {s['mean_placebo_bps']:+.1f} bps  "
          f"vs-March p = {s['pval_placebo_march']:.3f}\n")

    print("=== Day-of-March sweep (Bonferroni over 5 anchor days) ===")
    print(s["mc_table"].to_string(index=False))
    print(f"\n  St. Patrick (17th) Bonferroni p = {s['pval_bonferroni_17']:.3f}\n")

    print("=== Sub-period split ===")
    print(s["subperiod_table"].to_string(index=False))
    print()

    print("=== Cost-aware strategy (long index on the St. Patrick session only) ===")
    print(f"  Net mean: {s['net_mean_bps']:+.1f} bps/event  HAC t = {s['net_tstat_hac']:+.2f}")
    print(f"  Per-event compounded: {s['strat_per_event_pct']:+.4f}%\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE — "
          "a one-regime bump dressed as a holiday rally.")


if __name__ == "__main__":
    main()
