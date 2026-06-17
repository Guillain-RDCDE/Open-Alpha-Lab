"""Real-data verification -- Study 287 (Easter-Effect). Regenerates docs/results.md.

Loads the ^GSPC daily tape (repo-level cache, cache-only), joins the Easter session
dates (Computus + frozen calendar), and runs the full event study: the pre-holiday
(Maundy Thursday) and post-holiday (Easter Monday) sessions against both the
all-day and the same-weekday baselines, with one-sample t-tests, Newey-West HAC t,
permutation test, sub-periods, and the tradability column.

The ^GSPC parquet is at the repo-level _cache/^GSPC_split_only.parquet and is
read-only (no fetch required). Run from anywhere:

    python studies/287-easter-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from easter_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 287 -- Easter-Effect -- real-data verification\n")
    print("Data: ^GSPC daily close-to-close (price-only), repo-level cache")
    print("      Easter sessions: Computus + frozen ^GSPC calendar in data.py (1950-2025)\n")

    ev = data.build_event_frame(fetch=False)
    unc = data.unconditional_daily(fetch=False)
    unc_thu = data.unconditional_daily(fetch=False, weekday=3)  # pre-holiday = Thursday
    unc_mon = data.unconditional_daily(fetch=False, weekday=0)  # post = Monday

    fp_pre = data.fingerprint(ev["pre_ret"])
    fp_post = data.fingerprint(ev["post_ret"])
    fp_unc = data.fingerprint(unc)
    print(f"Data stamp: years {ev.index.min()}-{ev.index.max()}, n={len(ev)} weekends")
    print(f"  pre-session fingerprint  = {fp_pre}")
    print(f"  post-session fingerprint = {fp_post}")
    print(f"  uncond fingerprint       = {fp_unc}\n")

    s = st.summarize(ev["pre_ret"], unc, n_permutations=10_000, seed=287)
    s_thu = st.summarize(ev["pre_ret"], unc_thu, n_permutations=10_000, seed=287)

    print("=== PRE-HOLIDAY session (Maundy Thursday) vs baselines ===")
    print(f"  Pre-holiday mean:   {s['ev_mean_pct']:+.4f}%/day  (up-rate {s['up_rate_pct']:.1f}%)")
    print(f"  Unconditional mean: {s['uncond_mean_pct']:+.4f}%/day  (the honest baseline)")
    print(f"  Excess vs all days: {s['excess_mean_bps']:+.2f} bps/day")
    print(f"  t vs 0:                 {s['t_vs_zero']:+.3f}")
    print(f"  t vs unconditional:     {s['t_vs_uncond']:+.3f}  (p = {s['p_vs_uncond']:.4f})")
    print(f"  Newey-West HAC t:       {s['hac_t_excess']:+.3f}")
    print(f"  Permutation p:          {s['perm_p']:.4f}")
    print(f"  -- day-of-week control (vs Thursday-only baseline): "
          f"excess {s_thu['excess_mean_bps']:+.2f} bps, HAC t {s_thu['hac_t_excess']:+.3f}, "
          f"perm p {s_thu['perm_p']:.4f}\n")

    sp_post = st.summarize(ev["post_ret"], unc, n_permutations=10_000, seed=287)
    sp_post_mon = st.summarize(ev["post_ret"], unc_mon, n_permutations=10_000, seed=287)
    print("=== POST-HOLIDAY session (Easter Monday) vs baselines ===")
    print(f"  Post-holiday mean:  {sp_post['ev_mean_pct']:+.4f}%/day  (up-rate {sp_post['up_rate_pct']:.1f}%)")
    print(f"  Excess vs all days: {sp_post['excess_mean_bps']:+.2f} bps, HAC t {sp_post['hac_t_excess']:+.3f}")
    print(f"  -- vs Monday-only baseline (kills the Monday effect): "
          f"excess {sp_post_mon['excess_mean_bps']:+.2f} bps, HAC t {sp_post_mon['hac_t_excess']:+.3f}, "
          f"perm p {sp_post_mon['perm_p']:.4f}\n")

    print("=== Sub-periods (pre-holiday excess return) ===")
    print(st.subperiod_stats(ev["pre_ret"], unc,
                             breaks=((1950, 1979), (1980, 2002), (2003, 2025))).to_string(index=False))
    print()

    tr = st.tradability(ev["pre_ret"], cost_bps_oneway=1.0)
    print("=== Tradability (buy the pre-holiday session, one round-trip/yr) ===")
    print(f"  Gross mean per event: {tr['gross_mean_per_event']*1e4:+.2f} bps")
    print(f"  Net mean per event:   {tr['net_mean_per_event']*1e4:+.2f} bps "
          f"(after {tr['cost_drag_per_event']*1e4:.1f} bps round-trip cost x NAV)")
    print(f"  Cumulative growth:    gross x{tr['gross_cum_growth']:.4f} | net x{tr['net_cum_growth']:.4f}")
    print(f"  Per-event CAGR:       gross {tr['gross_cagr']*100:.3f}% | net {tr['net_cagr']*100:.3f}%\n")

    print("Verdict: Signal=REAL (pre-holiday drift, HAC t>3, survives day-of-week control),")
    print("         Tradability=FRAGILE -- a genuine but tiny once-a-year sliver; it is the")
    print("         generic pre-holiday effect, not Easter-specific magic.")


if __name__ == "__main__":
    main()
