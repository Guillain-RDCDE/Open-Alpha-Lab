"""Reproducible headline run for Study 604 — Month-End Rebalancing Flows.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY/AGG/VBMFX tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from month_end_rebalancing_flows import data, strategy as st

print("# Month-End Rebalancing Flows — SPY vs the spliced Aggregate-bond leg "
      "(VBMFX -> AGG at 2003-09-30), yfinance total-return")

if data.have_real():
    try:
        from quantlab import repro
        px = data.load_prices()
        print(repro.data_stamp("SPY/AGG/VBMFX total-return closes", px, asof=data.AS_OF))
    except ImportError:
        pass
    rets = data.load_real()
    tab = st.month_table(rets)
    print(f"daily returns   : {len(rets):,} sessions, {rets.index.min().date()} -> "
          f"{rets.index.max().date()} (bond leg = VBMFX before {data.SPLICE_DATE}, "
          f"AGG after — same Aggregate index, spliced in RETURN space)")
    print(f"month table     : {len(tab)} complete months, {tab.index.min()} -> "
          f"{tab.index.max()}; windows = last {st.WINDOW} / first {st.WINDOW} "
          f"trading days")

    print("\n# Leg 1 — the SELL-THE-WINNER leg: last-3-day eq-minus-bond spread, "
          "conditioned on the MTD gap through the close 3 days before month-end")
    a = st.quintile_split(tab, "gap_pre", "last_spread")
    print(f"  top-gap quintile   : {a['top_mean_bps']:+.2f} bps over the last 3 days "
          f"(one-sample t = {a['t_top']:+.2f}, n = {a['n_top']})")
    print(f"  bottom-gap quintile: {a['bot_mean_bps']:+.2f} bps (t = {a['t_bot']:+.2f}, "
          f"n = {a['n_bot']})")
    print(f"  unconditional      : {a['uncond_mean_bps']:+.2f} bps "
          f"(t = {a['t_uncond']:+.2f}, n = {a['n']})")
    print(f"  top-minus-bottom   : {a['diff_bps']:+.2f} bps   Welch t = "
          f"{a['welch_t']:+.2f} | top-vs-unconditional Welch t = "
          f"{a['top_vs_uncond_t']:+.2f}")
    p1 = st.placebo_pvalue(tab, "gap_pre", "last_spread", side="le")
    print(f"  permutation placebo (gap shuffled across months, "
          f"{p1['n_seeds']} seeds x {p1['n_draws']:,} draws): p = {p1['p_mean']:.4f} "
          f"[{p1['p_min']:.4f}, {p1['p_max']:.4f}]")

    print("\n# Leg 2 — the REVERSAL leg: first-3-day spread of the NEXT month, "
          "conditioned on the full prior-month gap")
    b = st.quintile_split(tab, "gap_full", "first_spread")
    print(f"  top-gap quintile   : {b['top_mean_bps']:+.2f} bps "
          f"(one-sample t = {b['t_top']:+.2f}, n = {b['n_top']})")
    print(f"  bottom-gap quintile: {b['bot_mean_bps']:+.2f} bps (t = {b['t_bot']:+.2f})")
    print(f"  unconditional      : {b['uncond_mean_bps']:+.2f} bps "
          f"(t = {b['t_uncond']:+.2f}, n = {b['n']})")
    print(f"  top-minus-bottom   : {b['diff_bps']:+.2f} bps   Welch t = "
          f"{b['welch_t']:+.2f}")
    p2 = st.placebo_pvalue(tab, "gap_full", "first_spread", side="ge")
    print(f"  permutation placebo: p = {p2['p_mean']:.4f} "
          f"[{p2['p_min']:.4f}, {p2['p_max']:.4f}]")

    print("\n# Dose-response — mean window spread (bps) per conditioning quintile")
    qa = st.quintile_profile(tab, "gap_pre", "last_spread")
    qb = st.quintile_profile(tab, "gap_full", "first_spread")
    for q in qa.index:
        print(f"  Q{q}: gap {qa.loc[q,'cond_mean_pct']:+.2f}% -> last-3 spread "
              f"{qa.loc[q,'mean_bps']:+.2f} bps (t = {qa.loc[q,'t']:+.2f}) | "
              f"first-3 spread {qb.loc[q,'mean_bps']:+.2f} bps "
              f"(t = {qb.loc[q,'t']:+.2f})")

    print("\n# Era split (the splice era / ETF era / recent)")
    for e in st.era_split(tab, "gap_pre", "last_spread"):
        print(f"  SELL {e['era']:>20}: top {e['top_mean_bps']:+.2f} bps "
              f"(t = {e['t_top']:+.2f}) | diff {e['diff_bps']:+.2f} bps, "
              f"Welch t = {e['welch_t']:+.2f} (n = {e['n']})")
    for e in st.era_split(tab, "gap_full", "first_spread"):
        print(f"  REV  {e['era']:>20}: top {e['top_mean_bps']:+.2f} bps "
              f"(t = {e['t_top']:+.2f}) | diff {e['diff_bps']:+.2f} bps, "
              f"Welch t = {e['welch_t']:+.2f} (n = {e['n']})")

    print("\n# Window robustness — the last-k / first-k window length")
    for w in (2, 3, 4):
        t2 = st.month_table(rets, window=w)
        aw = st.quintile_split(t2, "gap_pre", "last_spread")
        bw = st.quintile_split(t2, "gap_full", "first_spread")
        print(f"  k={w}: SELL top {aw['top_mean_bps']:+.2f} bps "
              f"(t = {aw['t_top']:+.2f}), diff Welch t = {aw['welch_t']:+.2f} | "
              f"REV top {bw['top_mean_bps']:+.2f} bps (t = {bw['t_top']:+.2f}), "
              f"diff Welch t = {bw['welch_t']:+.2f}")

    print("\n# Tradability — expanding-quintile flow trade (short eq/long bd last 3 "
          "days, flip long eq/short bd first 3 days; 60-month burn-in, 8 one-way "
          "tickets/event, short leg pays 30 bps/yr borrow)")
    for cb in (2.0, 5.0):
        tr = st.flow_trade(rets, cost_bps=cb)
        print(f"  cost {cb:>3.0f} bps/leg: {tr['n_events']} events, "
              f"{tr['share_active']*100:.1f}% of days active | gross "
              f"{tr['gross_ann_pct']:+.2f}%/yr -> net {tr['net_ann_pct']:+.2f}%/yr | "
              f"per-event gross {tr['per_event_gross_bps']:+.1f} bps vs "
              f"{tr['per_event_cost_bps']:.0f} bps tickets | HAC t (net, active days) "
              f"= {tr['t_net_active']:+.2f}")
    for lo, hi, tag, burn in [(None, "2015-12-31", "1993-2015", 60),
                              ("2016-01-01", None, "2016-2026", 24)]:
        r2 = rets
        if lo:
            r2 = r2[r2.index >= lo]
        if hi:
            r2 = r2[r2.index <= hi]
        tr = st.flow_trade(r2, cost_bps=2.0, burn_in=burn)
        print(f"  sub-period {tag} (2 bps/leg, burn-in {burn}m): {tr['n_events']} "
              f"events | net {tr['net_ann_pct']:+.2f}%/yr | HAC t = "
              f"{tr['t_net_active']:+.2f}")

    print("\n# Third axis — is the turn-of-the-month (study 89) just rebalancing "
          "flows in disguise?")
    tv = st.tom_vs_flows(tab)
    print(f"  unconditional first-3-day SPY drift : {tv['uncond_bps']:+.2f} bps "
          f"(t = {tv['t_uncond']:+.2f})")
    print(f"  middle 3 gap-quintiles (no pressure): {tv['mid_bps']:+.2f} bps "
          f"(t = {tv['t_mid']:+.2f}, n = {tv['n_mid']})")
    print(f"  top gap quintile                    : {tv['top_bps']:+.2f} bps "
          f"(t = {tv['t_top']:+.2f}) | bottom: {tv['bot_bps']:+.2f} bps "
          f"(t = {tv['t_bot']:+.2f})")
    print(f"  extremes-vs-middle Welch t          : {tv['welch_ext_vs_mid']:+.2f} "
          f"(n_ext = {tv['n_ext']})")
else:
    print("(no _cache/merf_prices.csv — run data.fetch_tape() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must stay quiet on a pure two-asset random walk and light up "
      "when a flow reversal is planted.")
null_ws = []
for s in range(20):
    w0 = data.synthetic_world(edge=0.0, seed=s)
    t0 = st.month_table(w0)
    null_ws.append(st.quintile_split(t0, "gap_pre", "last_spread")["welch_t"])
null_ws = np.asarray(null_ws)
print(f"  null (edge = 0, 20 seeds): SELL diff Welch t mean {null_ws.mean():+.2f}, "
      f"sd {null_ws.std(ddof=1):.2f}, max |t| = {np.abs(null_ws).max():.2f} "
      f"(0 of 20 clear |t| >= 2: {int((np.abs(null_ws) >= 2).sum())} exceed)")
w1 = data.synthetic_world(edge=0.010, seed=604)
t1 = st.month_table(w1)
a1 = st.quintile_split(t1, "gap_pre", "last_spread")
b1 = st.quintile_split(t1, "gap_full", "first_spread")
tr1 = st.flow_trade(w1, cost_bps=2.0)
w0 = data.synthetic_world(edge=0.0, seed=604)
tr0 = st.flow_trade(w0, cost_bps=2.0)
print(f"  planted 100 bps reversal (seed 604): SELL diff Welch t = "
      f"{a1['welch_t']:+.2f} | REV diff Welch t = {b1['welch_t']:+.2f} | flow trade "
      f"net {tr1['net_ann_pct']:+.2f}%/yr, HAC t = {tr1['t_net_active']:+.2f}")
print(f"  null trade (edge = 0, seed 604): net {tr0['net_ann_pct']:+.2f}%/yr, "
      f"HAC t = {tr0['t_net_active']:+.2f}")
