"""Reproducible headline run for Study 603 — Treasury Auction Concession.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached auction records + yield tape +
TLT under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from treasury_auction_concession import data, strategy as st

print("# Treasury Auction Concession — 10Y/30Y auctions (TreasuryDirect/FiscalData) vs "
      "^TNX/^TYX daily yields + TLT")

if data.have_real():
    try:
        from quantlab import repro
        au, ylds, tlt = data.load_real()
        print(repro.data_stamp("^TNX/^TYX daily yields (%)", ylds, asof=data.AS_OF))
        print(repro.data_stamp("TLT total-return close + ^IRX", tlt, asof=data.AS_OF))
    except ImportError:
        au, ylds, tlt = data.load_real()
    n10 = int((au["tenor"] == "10Y").sum())
    n30 = int((au["tenor"] == "30Y").sum())
    print(f"auctions        : {len(au)} ({n10} x 10Y notes, {n30} x 30Y bonds), "
          f"{au['auction_date'].min().date()} -> {au['auction_date'].max().date()}, "
          f"reopenings included (same supply event)")

    print("\n# The concession — daily yield changes around auctions "
          "(pre = 5 trading days ending on the auction-day close; post = the 5 after)")
    S = {}
    for tenor, col in [("10Y", "y10"), ("30Y", "y30")]:
        s = st.summarize_tenor(ylds[col], au, tenor)
        S[tenor] = s
        r = s["reg"]
        print(f"  {tenor}: {s['n_auctions']} auctions on the tape")
        print(f"    event windows : pre {s['pre_mean_bps']:+.2f} bps cum | "
              f"post {s['post_mean_bps']:+.2f} bps cum | ordinary 5-day baseline "
              f"{s['base_mean_bps']:+.2f} bps (n={s['n_base']:,}) | "
              f"Welch pre-vs-baseline t = {s['welch_pre_vs_base']:+.2f} (supporting)")
        print(f"    HAC regression (PRIMARY, NW lags {st.NW_LAGS}): "
              f"pre {r['pre_bps_day']:+.3f} bps/day (window {r['pre_bps_day']*5:+.2f} bps) "
              f"t = {r['t_pre']:+.2f} | post {r['post_bps_day']:+.3f} bps/day "
              f"(window {r['post_bps_day']*5:+.2f} bps) t = {r['t_post']:+.2f} "
              f"| {r['n_days']:,} days")

    print("\n# Third axis — do BIGGER auctions get bigger concessions? "
          "(size detrended by the trailing median of the prior 8 same-tenor auctions)")
    for tenor, col in [("10Y", "y10"), ("30Y", "y30")]:
        ss = st.size_split(au, ylds[col], tenor)
        print(f"  {tenor}: big pre {ss['pre_big_bps']:+.2f} bps (n={ss['n_big']}) vs "
              f"small pre {ss['pre_small_bps']:+.2f} bps (n={ss['n_small']}) | "
              f"Welch t = {ss['welch_t_pre']:+.2f} | post: big {ss['post_big_bps']:+.2f} "
              f"vs small {ss['post_small_bps']:+.2f}, Welch t = {ss['welch_t_post']:+.2f}")

    print("\n# Era split — pre-2008 / 2008-2019 / 2020+ supply flood (HAC t per era)")
    for tenor, col in [("10Y", "y10"), ("30Y", "y30")]:
        for e in st.era_split(ylds[col], au.loc[au["tenor"] == tenor, "auction_date"]):
            print(f"  {tenor} {e['era']}: pre {e['pre_bps_day']:+.3f} bps/day "
                  f"t = {e['t_pre']:+.2f} | post {e['post_bps_day']:+.3f} bps/day "
                  f"t = {e['t_post']:+.2f} | {e['n_auctions']} auctions")

    print("\n# Tradability — long TLT over the post-auction window "
          "(entry at the auction-day close, hold 5 days; excess of 13-week bills)")
    dates_all = au["auction_date"]
    al = st.tlt_post_alpha(tlt, dates_all)
    print(f"  TLT buy & hold excess of bills : {al['bh_excess_ann_pct']:+.2f}%/yr "
          f"({al['n_days']:,} days) — the beta yardstick")
    print(f"  post-auction ALPHA (HAC reg)   : ordinary day {al['base_bps_day']:+.2f} bps/day, "
          f"post-auction day +{al['post_extra_bps_day']:.2f} bps/day extra, "
          f"t = {al['t_post_extra']:+.2f} ({al['n_post_days']:,} post days)")
    for cb in (2.0, 5.0):
        tr = st.tlt_post_auction(tlt, dates_all, cost_bps=cb)
        print(f"  cost {cb:>3.0f} bps/leg: {tr['n_round_trips']} round trips, in-market "
              f"{tr['held_share']*100:.1f}% of days | gross {tr['gross_ann_pct']:+.2f}%/yr "
              f"-> net {tr['net_ann_pct']:+.2f}%/yr | per-trip gross "
              f"{tr['per_trip_gross_bps']:+.1f} bps vs {tr['per_trip_cost_bps']:.0f} bps "
              f"round-trip cost | HAC t (net, held days) = {tr['t_net_held']:+.2f}")
    for lo, hi, tag in [(None, "2019-12-31", "2002-2019"), ("2020-01-01", None, "2020-2026")]:
        t2 = tlt
        d2 = dates_all
        if lo:
            t2 = t2[t2.index >= lo]
            d2 = d2[d2 >= lo]
        if hi:
            t2 = t2[t2.index <= hi]
            d2 = d2[d2 <= hi]
        tr = st.tlt_post_auction(t2, d2, cost_bps=2.0)
        print(f"  sub-period {tag} (2 bps/leg): {tr['n_round_trips']} trips | "
              f"gross {tr['gross_ann_pct']:+.2f}%/yr -> net {tr['net_ann_pct']:+.2f}%/yr | "
              f"HAC t = {tr['t_net_held']:+.2f}")
else:
    print("(no _cache/ — run data.fetch_auctions() + data.fetch_tape() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must stay quiet on a pure random-walk null and must light up when a")
print("  concession/richening V is planted around the synthetic auctions.")
for c, r, label in [(0.0, 0.0, "null (0/0)"), (4.0, 4.0, "planted 4 bps in / 4 bps out")]:
    au_s, y_s, tlt_s = data.synthetic_world(concession_bps=c, richen_bps=r, seed=603)
    reg = st.dummy_regression(y_s["y10"], au_s["auction_date"])
    tr = st.tlt_post_auction(tlt_s, au_s["auction_date"], cost_bps=2.0)
    print(f"  {label}: pre {reg['pre_bps_day']:+.3f} bps/day t = {reg['t_pre']:+.2f} | "
          f"post {reg['post_bps_day']:+.3f} bps/day t = {reg['t_post']:+.2f} | "
          f"fund trade net {tr['net_ann_pct']:+.2f}%/yr t = {tr['t_net_held']:+.2f}")
