"""Reproducible headline run for Study 636 — Exchange Listing Pop.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached Binance close panel under
``_cache/`` (the real-tape numbers) and always runs the synthetic control offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exchange_listing_pop import data, strategy as st

WINDOWS = [
    ("early run-up  [-20..-6]", -20, -6),
    ("run-up        [-5..-1] ", -5, -1),
    ("listing day   [0]      ", 0, 0),
    ("THE POP       [-5..0]  ", -5, 0),
    ("post          [+1..+5] ", 1, 5),
    ("THE FADE      [+1..+30]", 1, 30),
]

print("# Exchange Listing Pop — Coinbase listing events priced on the Binance tape")
if data.have_real():
    px = data.load_real()
    ev = data.events_frame()
    ev = ev[ev["coin"].isin(px.columns)].reset_index(drop=True)
    abn = st.abnormal_returns(px)
    span = (px.index.max() - px.index.min()).days / 365.25
    print(f"panel          : {px.shape[0]} days x {px.shape[1]} coins (incl BTC), "
          f"{px.index.min().date()} -> {px.index.max().date()} ({span:.1f} years)")
    # events actually usable (full window coverage) — count via the widest window
    usable = st.event_car(abn, ev, -20, 30)
    n_cl = usable["day0"].nunique()
    pre = usable[usable["day0"] < data.ROADMAP_ERA]
    post = usable[usable["day0"] >= data.ROADMAP_ERA]
    print(f"events         : {len(ev)} listings hardcoded; {len(usable)} with full "
          f"[-20..+30] coverage, in {n_cl} same-day clusters")
    print(f"era split      : {len(pre)} events pre-roadmap (< {data.ROADMAP_ERA.date()}), "
          f"{len(post)} in the roadmap era")
    print(f"fingerprint    : {data.fingerprint(px)}")

    print("\n# Event windows — BTC-adjusted CARs (log, cluster-level one-sample t)")
    W = {}
    for label, a, b in WINDOWS:
        s = st.window_summary(abn, ev, a, b)
        W[(a, b)] = s
        print(f"  {label}: mean CAR {s['mean_car']*100:+6.2f}% (simple "
              f"{s['mean_simple_pct']:+6.2f}%)  t = {s['t']:+5.2f}  "
              f"(n = {s['n_clusters']} clusters / {s['n_events']} events)")

    print("\n# Inference on the two headline windows (placebo = random dates, same coins)")
    for label, a, b in [("POP  [-5..0]", -5, 0), ("FADE [+1..+30]", 1, 30)]:
        side = "greater" if a < 1 else "less"
        pl = st.placebo_pvalue(abn, ev, a, b, n_draws=2000, seed=636, side=side)
        # seed-robustness of the empirical p (>= 20 seeds, desk rule for random baselines)
        ps = [st.placebo_pvalue(abn, ev, a, b, n_draws=500, seed=s0, side=side)["p"]
              for s0 in range(1, 21)]
        print(f"  {label}: obs event-level mean CAR {pl['obs']*100:+.2f}%  Welch t (events vs "
              f"{pl['n_pooled']:,} placebo windows) = {pl['welch_t']:+.2f}  "
              f"placebo p = {pl['p']:.4f}  (20-seed p mean {np.mean(ps):.4f}, "
              f"range {min(ps):.4f}-{max(ps):.4f})")

    print("\n# Robustness — coarser clustering + hit rates (are a few outliers driving it?)")
    for label, a, b in [("POP  [-5..0]", -5, 0), ("FADE [+1..+30]", 1, 30)]:
        cars = st.event_car(abn, ev, a, b)
        cl = st.cluster_by_day(cars)
        bymonth = cars.assign(m=cars["day0"].dt.to_period("M")).groupby("m")["car"].mean()
        mm, tm, nm = st.one_sample_t(bymonth.to_numpy())
        med = float(np.median(cl.to_numpy()))
        pos = float((cl.to_numpy() > 0).mean())
        print(f"  {label}: month-clustered t = {tm:+5.2f} (n = {nm} months)  "
              f"median cluster CAR {med*100:+6.2f}%  share of clusters > 0: {pos*100:.0f}%")

    print("\n# Era split — before vs after the 2022-04-28 'listing roadmap' policy")
    ev_pre = ev[ev["day0"] < data.ROADMAP_ERA]
    ev_post = ev[ev["day0"] >= data.ROADMAP_ERA]
    for nm, sub in [("pre-roadmap ", ev_pre), ("roadmap era ", ev_post)]:
        for label, a, b in [("POP  [-5..0]", -5, 0), ("FADE [+1..+30]", 1, 30)]:
            s = st.window_summary(abn, sub, a, b)
            print(f"  {nm} {label}: mean CAR {s['mean_car']*100:+6.2f}%  "
                  f"t = {s['t']:+5.2f}  (n = {s['n_clusters']} clusters)")
    # Welch t of the pop difference between eras
    cpre = st.cluster_by_day(st.event_car(abn, ev_pre, -5, 0)).to_numpy()
    cpost = st.cluster_by_day(st.event_car(abn, ev_post, -5, 0)).to_numpy()
    print(f"  pop difference pre-vs-roadmap: Welch t = {st.welch_t(cpre, cpost):+.2f}")

    print("\n# Tradability — the follower's trade (enter at the day-0 close, ONE lag)")
    print("  long the new listing, hedged short-BTC, borrow 3%/yr on the hedge leg:")
    for hold in (5, 30):
        for cb in (10.0, 25.0, 50.0):
            tr = st.follower_trade(px, ev, hold=hold, cost_bps=cb, hedged=True, side="long")
            s = st.trade_summary(tr)
            print(f"    hold +{hold:>2}d cost {cb:>4.0f} bps: gross {s['gross_pct']:+6.2f}% "
                  f"(t {s['t_gross']:+5.2f}) -> net {s['net_pct']:+6.2f}% (t {s['t_net']:+5.2f}) "
                  f" hit {s['hit']*100:.0f}%  (n = {s['n']})")
    print("  FADE it: short the new listing, hedged long-BTC, borrow 10%/yr on the alt:")
    for hold in (30,):
        for cb in (10.0, 25.0, 50.0):
            tr = st.follower_trade(px, ev, hold=hold, cost_bps=cb, hedged=True, side="short")
            s = st.trade_summary(tr)
            print(f"    hold +{hold:>2}d cost {cb:>4.0f} bps: gross {s['gross_pct']:+6.2f}% "
                  f"(t {s['t_gross']:+5.2f}) -> net {s['net_pct']:+6.2f}% (t {s['t_net']:+5.2f}) "
                  f" hit {s['hit']*100:.0f}%  (n = {s['n']})")
    print("  borrow sensitivity on the fade short (25 bps one-way, hold +30d):")
    for br in (0.10, 0.50, 1.00):
        tr = st.follower_trade(px, ev, hold=30, cost_bps=25.0, hedged=True,
                               side="short", borrow_alt=br)
        s = st.trade_summary(tr)
        print(f"    alt borrow {br*100:>4.0f}%/yr: net {s['net_pct']:+6.2f}% "
              f"(t {s['t_net']:+5.2f})  hit {s['hit']*100:.0f}%")
    print("  reference (UNTRADABLE): you would have needed the news in advance —")
    frn = st.window_summary(abn, ev, -5, 0)
    print(f"    the pop itself, CAR[-5..0] = {frn['mean_car']*100:+.2f}% "
          f"(t = {frn['t']:+.2f}) accrues BEFORE the first close you can trade")

    print("\n# Third axis — announcement vs listing: is the tradable part gone?")
    print("  documented announcement -> first-trade gaps (Coinbase blog, per-row source in data.py):")
    for coin, ann, trade, src in data.ANNOUNCEMENTS:
        gap = (pd.Timestamp(trade) - pd.Timestamp(ann)).days
        print(f"    {coin:>6}: announced {ann}, trading {trade}  (gap {gap} days)")
    print("  2022-04-28+: the public 'listing roadmap' front-loads the news by weeks.")
    pop_car = st.window_summary(abn, ev, -5, 0)
    post_car = st.window_summary(abn, ev, 1, 30)
    print(f"  pop CAR[-5..0] = {pop_car['mean_car']*100:+.2f}% (t = {pop_car['t']:+.2f}) vs "
          f"post-listing CAR[+1..+30] = {post_car['mean_car']*100:+.2f}% "
          f"(t = {post_car['t']:+.2f})")
else:
    print("(no _cache/elp_prices.parquet — run data.fetch_panel(fetch=True) once)")

print("\n# Synthetic control — deterministic, no network")
print("  the event-study machinery must recover a PLANTED pop+fade and stay quiet on the null.")
for pop, fade in [(0.0, 0.0), (0.15, 0.10)]:
    spx, sev = data.synthetic_world(pop=pop, fade=fade, seed=636)
    sabn = st.abnormal_returns(spx)
    sp = st.window_summary(sabn, sev, -5, 0)
    sf = st.window_summary(sabn, sev, 1, 30)
    print(f"  planted pop={pop*100:5.1f}% fade={fade*100:5.1f}%: "
          f"CAR[-5..0] = {sp['mean_car']*100:+6.2f}% (t {sp['t']:+6.2f})  "
          f"CAR[+1..+30] = {sf['mean_car']*100:+6.2f}% (t {sf['t']:+6.2f})")
