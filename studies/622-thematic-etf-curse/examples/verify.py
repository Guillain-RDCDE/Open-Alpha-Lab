"""Reproducible headline run for Study 622 — Thematic ETF Curse.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panel under ``_cache/``
if present (the real-tape numbers) and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from thematic_etf_curse import data, strategy as st  # noqa: E402
from quantlab import repro  # noqa: E402

AS_OF = "2026-06-30"        # last complete month = June 2026

print("# Thematic ETF Curse — post-launch alpha of specialized ETF launches (yfinance)")
if data.have_real():
    px, rf = data.load_prices(), data.load_rf()
    pxa = repro.as_of(px, AS_OF)
    fp = repro.fingerprint(pxa)
    univ = data.build_universe(px, rf, as_of=AS_OF)
    th, br = univ["thematic"], univ["broad"]
    print(f"panel          : {len(th)} thematic + {len(br)} broad-index ETFs + SPY, "
          f"daily total-return closes {pxa.index.min().date()} -> {pxa.index.max().date()}")
    print(f"as-of {AS_OF}  fingerprint {fp}")
    lys = sorted(univ["launch"][t].year for t in th)
    print(f"thematic launches: {lys[0]}-{lys[-1]} (first candle = launch; "
          f"{sum(1 for t in th if univ['launch'][t].year >= 2018)} of {len(th)} in the 2018-2021 hype vintage)")

    print("\n# Calendar-time 'young thematics' portfolio — CAPM alpha vs SPY, Newey-West t (lags=6)")
    for w in (12, 36):
        r = st.young_alpha(univ, th, w=w)
        print(f"  first {w:>2} months : alpha {r['alpha_ann_pct']:+.2f}%/yr  HAC t = {r['t_alpha']:+.2f}  "
              f"beta {r['beta']:.2f}  ({r['n']} months, avg {r['avg_members']:.1f} members, "
              f"{r['start']} -> {r['end']})")
    rs = st.seasoned_alpha(univ, th)
    print(f"  seasoned (37+)  : alpha {rs['alpha_ann_pct']:+.2f}%/yr  HAC t = {rs['t_alpha']:+.2f}  "
          f"beta {rs['beta']:.2f}  ({rs['n']} months, avg {rs['avg_members']:.1f} members)")
    sp = st.young_vs_seasoned_spread(univ, th, w=36)
    print(f"  young-minus-seasoned spread alpha: {sp['alpha_ann_pct']:+.2f}%/yr  "
          f"HAC t = {sp['t_alpha']:+.2f}  (n={sp['n']}) — the launch-timing component per se")

    print("\n# Placebo — broad plain-vanilla index-ETF launches, same construction")
    rb = st.young_alpha(univ, br, w=36)
    print(f"  first 36 months : alpha {rb['alpha_ann_pct']:+.2f}%/yr  HAC t = {rb['t_alpha']:+.2f}  "
          f"beta {rb['beta']:.2f}  ({rb['n']} months)")

    print("\n# Event-time alpha curve — cumulative CAPM-abnormal return after launch")
    ec = st.event_curve(univ, th, k_max=48)
    for k in (12, 24, 36, 48):
        print(f"  month {k:>2}: cumulative {ec.loc[k, 'cum_ab_pct']:+.1f}%  "
              f"({int(ec.loc[k, 'n'])} funds contributing)")

    print("\n# Per-fund view — first-36-months CAPM alpha, one number per fund")
    pf = st.per_fund_alphas(univ, th, w=36)
    print(f"  median {pf['alpha_ann_pct'].median():+.2f}%/yr  "
          f"({(pf['alpha_ann_pct'] < 0).mean() * 100:.0f}% of {len(pf)} funds negative)")
    print("  (the calendar-time alpha is far worse than the median fund because launches CLUSTER")
    print("   ahead of the theme's bust — the months crowded with young funds were the bleed months)")

    print("\n# Robustness")
    r12 = st.young_alpha(univ, th, w=36, lags=12)
    r1 = st.young_alpha(univ, th, w=36, min_members=1)
    noark = [t for t in th if not t.startswith("ARK") and t != "IZRL"]
    rna = st.young_alpha(univ, noark, w=36)
    pre = [t for t in th if univ["launch"][t].year < 2018]
    post = [t for t in th if univ["launch"][t].year >= 2018]
    rpre = st.young_alpha(univ, pre, w=36)
    rpost = st.young_alpha(univ, post, w=36)
    print(f"  NW lags = 12          : t = {r12['t_alpha']:+.2f}")
    print(f"  min members = 1       : alpha {r1['alpha_ann_pct']:+.2f}%/yr  t = {r1['t_alpha']:+.2f}  ({r1['n']} months)")
    print(f"  ex-ARK ({len(noark)} funds)    : alpha {rna['alpha_ann_pct']:+.2f}%/yr  t = {rna['t_alpha']:+.2f}")
    print(f"  pre-2018 vintage ({len(pre):>2}) : alpha {rpre['alpha_ann_pct']:+.2f}%/yr  t = {rpre['t_alpha']:+.2f}  ({rpre['n']} months)")
    print(f"  2018+ vintage    ({len(post):>2}) : alpha {rpost['alpha_ann_pct']:+.2f}%/yr  t = {rpost['t_alpha']:+.2f}  ({rpost['n']} months)")

    print("\n# Tradability — short the young book, hedge beta x SPY (5 bps one-way, borrow swept)")
    for b in (300.0, 600.0, 1000.0):
        so = st.short_overlay(univ, th, w=36, cost_bps=5.0, borrow_bps_ann=b)
        tail = f"(gross {so['gross_ann_pct']:+.2f}%/yr t={so['t_gross']:+.2f})" if b == 300.0 else ""
        print(f"  borrow {b:>5.0f} bps/yr: net {so['net_ann_pct']:+.2f}%/yr  HAC t = {so['t_net']:+.2f}  {tail}")

    print("\n# Third axis — does buying the -50% dip in a thematic work?")
    dp = st.dip_portfolio(univ, th, dd=0.50, hold=36)
    f12 = np.array(dp["fwd12"])
    f36 = np.array(dp["fwd36"])
    print(f"  {dp['n_events']} funds hit a -50% drawdown; dip-buyer book (enter next month, hold 36m):")
    print(f"  CAPM alpha {dp['alpha_ann_pct']:+.2f}%/yr  HAC t = {dp['t_alpha']:+.2f}  "
          f"beta {dp['beta']:.2f}  ({dp['n']} months)")
    print(f"  forward 12m vs SPY: mean {f12.mean() * 100:+.1f} pp  "
          f"({(f12 < 0).mean() * 100:.0f}% of {len(f12)} events negative)")
    print(f"  forward 36m vs SPY: mean {f36.mean() * 100:+.1f} pp  "
          f"({(f36 < 0).mean() * 100:.0f}% of {len(f36)} events negative)")
else:
    print("(no _cache/tec_prices.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the calendar-time machinery must recover a PLANTED post-launch drag and must NOT")
print("  manufacture a negative alpha when the true drag is 0.")
for drag in (0.0, -0.08):
    u = data.synthetic_world(drag_ann=drag, seed=622)
    r = st.young_alpha(u, u["thematic"], w=36)
    print(f"  planted drag {drag * 100:+5.0f}%/yr: alpha {r['alpha_ann_pct']:+.2f}%/yr  "
          f"HAC t = {r['t_alpha']:+.2f}  ({r['n']} months)")
