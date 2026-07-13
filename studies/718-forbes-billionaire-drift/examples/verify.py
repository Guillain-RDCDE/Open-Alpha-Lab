"""Reproducible headline run for Study 718 — Forbes-Billionaire-Drift.

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

from forbes_billionaire_drift import data, strategy as st

print("# Forbes-Billionaire-Drift — short-window market-model CARs around the annual list")
if data.have_real():
    prices, events = data.load_real()
    dropped = [e["ticker"] for e in data.FORBES_EVENTS if e["ticker"] not in prices.columns]
    print(f"event vehicles cached : {sorted(c for c in prices.columns if c not in data.BENCHMARKS)}")
    print(f"events with price data: {len(events)} of {len(data.FORBES_EVENTS)} "
          f"(dropped, delisted: {dropped}) | table fingerprint {data.fingerprint(data.FORBES_EVENTS)}")

    print("\n# The three windows (market model, SPY benchmark)")
    for name, win in [("pre  [-63,-1]", st.PRE_WINDOW), ("announce [0,+2]", st.ANNOUNCE_WINDOW),
                      ("post [+1,+63]", st.POST_WINDOW)]:
        panel = st.car_panel(prices, events, window=win)
        b = st.summarize_bucket(panel["car"].to_numpy())
        pc = panel["car"].to_numpy(); pc = pc[np.isfinite(pc)]
        null = st.placebo_car_dist(prices, data.TICKERS, k=len(pc), window=win, n_draws=8000)
        p = st.placebo_pvalue(float(pc.mean()), null)
        print(f"  {name:16}: n={b['n']:>2}  mean={b['mean_pct']:+6.2f}%  "
              f"win={b['win']*100:>4.0f}%  t={b['t']:+5.2f}  placebo p={p:.3f}")

    print("\n# The post-list drift is a beta artifact, not alpha")
    for bench in ("SPY", "QQQ"):
        panel = st.car_panel(prices, events, window=st.POST_WINDOW, bench=bench)
        b = st.summarize_bucket(panel["car"].to_numpy())
        print(f"  market model vs {bench}: post mean={b['mean_pct']:+.2f}%  t={b['t']:+.2f}")
    rex = st.raw_excess_panel(prices, events, window=st.POST_WINDOW, bench="SPY")
    print(f"  plain excess over SPY (beta=1): post mean={rex.mean()*100:+.2f}%  "
          f"t={st.welch_t(rex):+.2f}   <- the drift dissolves")

    print("\n# The TRADABLE version — enter after publication, net of costs")
    for lag, lab in [(0, "[+1,+63]"), (1, "[+2,+64]")]:
        pl = st.car_panel(prices, events, window=st.POST_WINDOW, lag=lag)
        b = st.summarize_bucket(pl["car"].to_numpy())
        nc = st.net_of_costs(b["mean_pct"] / 100)
        print(f"  enter lag={lag} {lab}: mean={b['mean_pct']:+.2f}%  t={b['t']:+.2f}  "
              f"net@20bps={nc['net_pct']:+.2f}%")

    print("\n# Dispersion — per-name post-list CAR (a coin flip with fat tails)")
    p = st.car_panel(prices, events, window=st.POST_WINDOW).sort_values("car")
    for _, r in p.iterrows():
        print(f"  {r['ticker']:5} {r['list_year']}  {r['car']*100:+7.1f}%  {r['founder']}")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED post-list drift and must NOT manufacture")
print("  significance from ~25 ultra-high-vol events when the true edge is 0.")
for edge in (0.0, 3000.0):
    syn = data.synthetic_events(drift_bps=edge, seed=718)
    b = st.summarize_bucket(syn["post_car"])
    print(f"  planted drift={edge:+6.0f}bps: post mean={b['mean_pct']:+.2f}%  "
          f"t={b['t']:+.2f}  win={b['win']*100:.0f}%")
