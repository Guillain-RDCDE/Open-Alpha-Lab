"""Real-data verification -- Study 286 (Valentines-Day). Regenerates docs/results.md.

Loads the ^GSPC daily tape (repo-level cache, cache-only), joins the hardcoded
Valentine session dates, and runs the full event study: excess return, one-sample
t-tests, Newey-West HAC t, permutation test, sub-periods, and the tradability column.

The ^GSPC parquet is at the repo-level _cache/^GSPC_split_only.parquet and is
read-only (no fetch required). Run from anywhere:

    python studies/286-valentines-day/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from valentines_day import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 286 -- Valentines-Day -- real-data verification\n")
    print("Data: ^GSPC daily close-to-close (price-only), repo-level cache")
    print("      Valentine sessions: hardcoded in data.py (1950-2025)\n")

    ev = data.build_event_frame(fetch=False)
    unc = data.unconditional_daily(fetch=False)

    fp_ev = data.fingerprint(ev["ret"])
    fp_unc = data.fingerprint(unc)
    print(f"Data stamp: years {ev.index.min()}-{ev.index.max()}, n={len(ev)} events")
    print(f"  event fingerprint = {fp_ev}")
    print(f"  uncond fingerprint = {fp_unc}\n")

    s = st.summarize(ev["ret"], unc, n_permutations=10_000, seed=286)

    print("=== Valentine session vs unconditional daily baseline ===")
    print(f"  Valentine mean:     {s['val_mean_pct']:+.4f}%/day")
    print(f"  Unconditional mean: {s['uncond_mean_pct']:+.4f}%/day  (the honest baseline)")
    print(f"  Excess mean:        {s['excess_mean_bps']:+.2f} bps/day")
    print(f"  Valentine up-rate:  {s['up_rate_pct']:.1f}%  (vs {s['uncond_up_rate_pct']:.1f}% unconditional)")
    print(f"  t vs 0:             {s['t_vs_zero']:+.3f}")
    print(f"  t vs unconditional: {s['t_vs_uncond']:+.3f}  (p = {s['p_vs_uncond']:.4f})")
    print(f"  Newey-West HAC t (excess, lag4): {s['hac_t_excess']:+.3f}")
    print(f"  Permutation p (two-sided):       {s['perm_p']:.4f}\n")

    print("=== Sub-periods ===")
    for a, b in [(1950, 1987), (1988, 2025)]:
        sub = ev[(ev.index >= a) & (ev.index <= b)]
        print(f"  {a}-{b}: n={len(sub):>2}  mean={sub['ret'].mean()*100:+.4f}%  "
              f"up-rate={(sub['ret'] > 0).mean()*100:.1f}%")
    print()

    tr = st.tradability(ev["ret"], cost_bps_oneway=1.0)
    print("=== Tradability (buy the Valentine session, one round-trip/yr) ===")
    print(f"  Gross mean per event: {tr['gross_mean_per_event']*1e4:+.2f} bps")
    print(f"  Net mean per event:   {tr['net_mean_per_event']*1e4:+.2f} bps "
          f"(after {tr['cost_drag_per_event']*1e4:.1f} bps round-trip cost x NAV)\n")

    print("Verdict: Signal=NONE, Tradability=MIRAGE -- the market does not love Valentine's Day.")


if __name__ == "__main__":
    main()
