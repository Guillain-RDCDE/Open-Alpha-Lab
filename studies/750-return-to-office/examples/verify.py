"""Reproducible headline run for Study 750 — Return-to-Office.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily closes under
``_cache/`` if present (the real-tape event study), and always runs the synthetic
positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from return_to_office import data, strategy as st

print("# Return-to-Office — office-REIT basket reaction to big-employer RTO mandates")
print(f"office basket : {data.OFFICE_REITS}")
print(f"benchmarks    : {data.MARKET} (market model), {data.BROAD_REIT} (broad-REIT robustness)")
print(f"RTO calendar  : {len(data.RTO_EVENTS)} dated mandates; "
      f"{len(data.DELISTED)} office casualties DELISTED (survivorship, named)")

if data.have_real():
    prices, events = data.load_real()
    mem = data.members_present(prices)
    print(f"as-of         : {prices.index.max().date()}   "
          f"table fingerprint {data.fingerprint(events)}   basket members {len(mem)}")

    panel = st.car_panel(prices, events, mem)      # canonical CAR[0, +2] window
    print("\n# Per-event basket CAR[0,+2] (market model, SPY benchmark)")
    print(f"  {'date':>10} {'kind':>6} {'CAR':>8}  employer")
    for _, r in panel.iterrows():
        print(f"  {str(r['date'].date()):>10} {'strict' if r['strict'] else 'hybrid':>6} "
              f"{r['car']*100:>+7.2f}%  {r['employer'][:44]}")

    s = st.summarize(panel, prices=prices, members=mem, n_draws=8000)
    print("\n# Buckets (strict full-RTO vs hybrid) — CAR[0,+2]")
    for k in ("strict", "hybrid", "all"):
        b = s[k]
        print(f"  {k:>7}: n={b['n']:>2}  mean={b['mean_pct']:+6.3f}%  "
              f"win={b['win']*100:>4.0f}%  t(vs 0)={b['t']:+5.2f}")
    print(f"  strict - hybrid: {s['diff_pct']:+.3f}pp   Welch t = {s['diff_t']:+.2f}")
    print(f"  all-events placebo p (random basket windows) = {s['all_placebo_p']:.3f}  "
          f"(null mean {s['null_mean_pct']:+.3f}%)")

    print("\n# Robustness — vary the event window (SPY market model)")
    for w in [(0, 0), (0, 2), (-1, 1), (0, 4)]:
        pw = st.car_panel(prices, events, mem, window=w)
        sw = st.summarize(pw)
        print(f"  window {str(w):>8}: all mean={sw['all']['mean_pct']:+.3f}%  "
              f"t={sw['all']['t']:+.2f}  strict-hybrid={sw['diff_pct']:+.3f}pp  "
              f"diff t={sw['diff_t']:+.2f}")

    print("\n# Robustness — VNQ benchmark (does office react BEYOND all REITs that day?)")
    for w in [(0, 0), (0, 2)]:
        pw = st.car_panel(prices, events, mem, benchmark="VNQ", window=w)
        sw = st.summarize(pw)
        print(f"  window {str(w):>8} vs VNQ: all mean={sw['all']['mean_pct']:+.3f}%  "
              f"t={sw['all']['t']:+.2f}")

    print("\n# The TRADABLE version — enter 1 day after (lag=1), hold the window, cost 10 bps")
    for w, lab in [((0, 2), "[+1,+3]"), ((0, 4), "[+1,+5]")]:
        pl = st.car_panel(prices, events, mem, window=w, lag=1)
        al = st.summarize_bucket(pl["car"].to_numpy())
        nc = st.net_of_costs(al["mean_pct"] / 100)
        print(f"  lag1 {lab}: all mean={al['mean_pct']:+.3f}%  t={al['t']:+.2f}  "
              f"net@10bps={nc['net_pct']:+.3f}%")

    print("\n# Context PROXY (labelled, cited — not priced): Kastle 10-city office occupancy")
    ko = data.kastle_proxy()
    print(f"  Kastle Back-to-Work Barometer (Feb-2020=100): {ko.iloc[0]:.0f} "
          f"({ko.index[0].date()}) -> {ko.iloc[-1]:.0f} ({ko.index[-1].date()}) "
          f"— desks refilled to ~half, then plateaued")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED strict-bucket CAR edge and must NOT manufacture a")
print("  strict>hybrid gap from ~two dozen events when the true edge is 0.")
for edge in (0.0, 400.0):
    syn = data.synthetic_events(car_bps=edge, seed=750)
    sc = st.summarize_bucket(syn["strict_car"])
    hc = st.summarize_bucket(syn["hybrid_car"])
    dt = st.welch_t(syn["strict_car"], syn["hybrid_car"])
    diff = (np.mean(syn["strict_car"]) - np.mean(syn["hybrid_car"])) * 100
    print(f"  planted car_bps={edge:+6.0f}: strict mean={sc['mean_pct']:+.2f}% "
          f"t={sc['t']:+.2f}  hybrid mean={hc['mean_pct']:+.2f}%  "
          f"diff={diff:+.2f}pp  diff t={dt:+.2f}")
