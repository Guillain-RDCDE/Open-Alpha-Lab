"""Reproducible headline run for Study 616 — Muni-CEF-Tax-Loss ("Muni CEF Tax-Loss Season").

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached fund + benchmark prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with no
network. The exact seasonal placebo has NO random component at all (a full enumeration of the
132 ordered month pairs).

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from muni_cef_tax_loss import data, strategy as st

print("# Muni-CEF-Tax-Loss — 12-fund muni CEF panel vs MUB (yfinance, total-return)")
if data.have_real():
    import pandas as pd

    from quantlab import repro

    px = data.load_real()
    print(repro.data_stamp("muni CEF panel + MUB + VWLTX adjusted closes", px, asof=data.AS_OF))

    ex = data.monthly_excess(px, bench=data.BENCH)
    print(f"panel          : {ex.shape[1]} funds x {ex.shape[0]} months of MUB-excess, "
          f"{ex.index.min().date()} -> {ex.index.max().date()}")

    print("\n# Pooled fund-months (SECONDARY - cross-fund pseudo-replication overstates n)")
    pm = st.pooled_month_stats(ex)
    print(f"  December excess : {pm['dec_mean_bps']:+8.2f} bps/mo  (n={pm['n_dec']})   "
          f"Welch t vs rest = {pm['welch_dec_vs_rest']:+.2f}")
    print(f"  January excess  : {pm['jan_mean_bps']:+8.2f} bps/mo  (n={pm['n_jan']})   "
          f"Welch t vs rest = {pm['welch_jan_vs_rest']:+.2f}")
    print(f"  other months    : {pm['rest_mean_bps']:+8.2f} bps/mo  (n={pm['n_rest']:,})")
    print(f"  January vs December Welch t = {pm['welch_jan_vs_dec']:+.2f}")

    print("\n# PRIMARY - per-winter basket (one observation per winter, kills pseudo-replication)")
    wt = st.winter_table(ex)
    ws = st.winter_stats(wt)
    print(f"  winters         : {ws['n_winters']} (Dec {wt.index.min()} -> Jan {wt.index.max()+1})")
    print(f"  December excess : {ws['dec_mean_bps']:+8.2f} bps  one-sample t = {ws['t_dec']:+.2f}")
    print(f"  January excess  : {ws['jan_mean_bps']:+8.2f} bps  one-sample t = {ws['t_jan']:+.2f}"
          f"   hit rate {ws['hit_jan']*100:.1f}%")
    print(f"  snap (Jan-Dec)  : {ws['snap_mean_bps']:+8.2f} bps  one-sample t = {ws['t_snap']:+.2f}"
          f"   hit rate {ws['hit_snap']*100:.1f}%")
    print(f"  snap lag-1 autocorrelation = {ws['snap_ac1']:+.3f} (annual, non-overlapping obs)")

    print("\n# Exact seasonal placebo - all 132 ordered month pairs (deterministic, no RNG)")
    pl = st.exact_pair_placebo(ex)
    print(f"  observed Dec->Jan contrast : {pl['obs']*1e4:+.2f} bps")
    print(f"  rank among 132 ordered pairs = {pl['rank']}   exact p = {pl['p_value']:.4f} "
          f"(minimum attainable 1/132 = {1/132:.4f})")
    grid = st.adjacent_pair_grid(ex)
    print("  adjacent-pair grid (bps):",
          {k: round(v * 1e4, 1) for k, v in grid.items()})

    print("\n# Robustness - VWLTX benchmark (NAV-priced muni mutual fund, full 2000+ history)")
    ex2 = data.monthly_excess(px, bench=data.ALT_BENCH)
    wt2 = st.winter_table(ex2)
    ws2 = st.winter_stats(wt2)
    pl2 = st.exact_pair_placebo(ex2)
    print(f"  winters         : {ws2['n_winters']}")
    print(f"  January excess  : {ws2['jan_mean_bps']:+8.2f} bps  t = {ws2['t_jan']:+.2f}"
          f"   hit rate {ws2['hit_jan']*100:.1f}%")
    print(f"  December excess : {ws2['dec_mean_bps']:+8.2f} bps  t = {ws2['t_dec']:+.2f}")
    print(f"  snap (Jan-Dec)  : {ws2['snap_mean_bps']:+8.2f} bps  t = {ws2['t_snap']:+.2f}")
    print(f"  exact placebo p : {pl2['p_value']:.4f} (rank {pl2['rank']}/132)")

    print("\n# Sub-period fade (split at the sample midpoint - not snooped)")
    a = wt.loc[wt.index <= 2016]
    b = wt.loc[wt.index > 2016]
    print(f"  MUB   2007-2016 Jan: {a['jan'].mean()*1e4:+8.2f} bps  t = "
          f"{st.ttest_vs_zero(a['jan']):+.2f}  (n={len(a)})")
    print(f"  MUB   2017-2025 Jan: {b['jan'].mean()*1e4:+8.2f} bps  t = "
          f"{st.ttest_vs_zero(b['jan']):+.2f}  (n={len(b)})")
    print(f"  Welch t of the DIFFERENCE (early - late) = {st.welch_t(a['jan'], b['jan']):+.2f} "
          f"(fade not itself certified)")
    a2 = wt2.loc[wt2.index <= 2012]
    b2 = wt2.loc[wt2.index > 2012]
    print(f"  VWLTX 2000-2012 Jan: {a2['jan'].mean()*1e4:+8.2f} bps  t = "
          f"{st.ttest_vs_zero(a2['jan']):+.2f}  (n={len(a2)})")
    print(f"  VWLTX 2013-2025 Jan: {b2['jan'].mean()*1e4:+8.2f} bps  t = "
          f"{st.ttest_vs_zero(b2['jan']):+.2f}  (n={len(b2)})")

    print("\n# Tradable January swap - MUB -> basket at Dec close, back at Jan close (4 one-way legs)")
    for cb in (5.0, 10.0, 25.0):
        js = st.january_swap(wt, cost_bps=cb)
        print(f"  cost={cb:>4.1f} bps/leg (drag {js['drag_bps']:.0f} bps): gross "
              f"{js['gross_mean_bps']:+.2f} -> net {js['net_mean_bps']:+.2f} bps/winter   "
              f"t = {js['t_net']:+.2f}   hit {js['hit_net']*100:.1f}%")

    print("\n# Third axis - does buying Dec-15 beat buying Jan-15? (common exit: last Feb close)")
    dj = st.dec15_vs_jan15(px, data.BENCH, data.FUNDS)
    ds = st.dec15_stats(dj)
    print(f"  winters          : {ds['n_winters']}")
    print(f"  Dec-15 entry     : {ds['dec15_mean_bps']:+8.2f} bps excess to end-Feb")
    print(f"  Jan-15 entry     : {ds['jan15_mean_bps']:+8.2f} bps excess to end-Feb")
    print(f"  paired difference: {ds['diff_mean_bps']:+8.2f} bps  t = {ds['t_diff']:+.2f}   "
          f"hit {ds['hit_diff']*100:.1f}%")
    d_ex08 = dj.loc[dj.index != 2008, "diff"]
    print(f"  ex-2008 (the outlier winter): {d_ex08.mean()*1e4:+.2f} bps  t = "
          f"{st.ttest_vs_zero(d_ex08):+.2f}  (n={len(d_ex08)})")
else:
    print("(no _cache/mct_prices.csv - run data.fetch_panel() once to build the cache)")

print("\n# Synthetic control - deterministic, no network (machinery proof only)")
print("  the per-winter basket + exact placebo must NOT fire on a seasonally dead panel and")
print("  MUST light up when a December dump + January snap are planted.")
for dump, snap in ((0.0, 0.0), (0.010, 0.010)):
    exs = data.synthetic_excess(dump=dump, snap=snap, seed=616)
    wts = st.winter_table(exs)
    wss = st.winter_stats(wts)
    pls = st.exact_pair_placebo(exs)
    print(f"  planted dump={dump*1e4:>6.0f} bps snap={snap*1e4:>6.0f} bps:  "
          f"Jan {wss['jan_mean_bps']:+7.1f} bps t={wss['t_jan']:+5.2f}   "
          f"snap {wss['snap_mean_bps']:+7.1f} bps t={wss['t_snap']:+5.2f}   "
          f"placebo p={pls['p_value']:.4f}")
