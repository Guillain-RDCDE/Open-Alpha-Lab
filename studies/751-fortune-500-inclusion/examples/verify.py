"""Reproducible headline run for Study 751 — Fortune-500-Inclusion.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily closes under ``_cache/``
if present (the real-tape event study), and always runs the synthetic positive control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fortune_500_inclusion import data, strategy as st

print("# Fortune-500-Inclusion — short-window market-model abnormal returns around the list reveal")
if data.have_real():
    prices, events = data.load_real()
    n_add = sum(e["added"] for e in events)
    print(f"event tickers cached : {sorted(c for c in prices.columns if c != 'SPY')}")
    print(f"events with price data: {len(events)} of {len(data.FORTUNE_EVENTS)} "
          f"({n_add} added / {len(events) - n_add} dropped; "
          f"table fingerprint {data.fingerprint(data.FORTUNE_EVENTS)})")

    panel = st.car_panel(prices, events)          # canonical CAR[0, +2] window
    print("\n# Canonical event window CAR[0,+2] (market model, SPY benchmark)")
    print(panel.to_string(index=False,
                          formatters={"car": lambda v: f"{v*100:+.2f}%"}))

    s = st.summarize(panel, prices=prices, tickers=data.TICKERS, n_draws=8000)
    print("\n# Buckets (added vs dropped) — CAR[0,+2]")
    for k in ("added", "dropped", "all"):
        b = s[k]
        print(f"  {k:>8}: n={b['n']:>2}  mean={b['mean_pct']:+6.2f}%  "
              f"win={b['win']*100:>4.0f}%  t(vs 0)={b['t']:+5.2f}")
    print(f"  added - dropped: {s['diff_pct']:+.2f}pp   Welch t = {s['diff_t']:+.2f}")
    print(f"  added placebo p (random non-event windows) = {s['added_placebo_p']:.3f}")

    print("\n# The reveal-day repricing you CANNOT trade (window [0,0])")
    p00 = st.car_panel(prices, events, window=(0, 0))
    a00 = st.summarize_bucket(p00.loc[p00["added"], "car"].to_numpy())
    d00 = st.welch_t(p00.loc[p00["added"], "car"].to_numpy(),
                     p00.loc[~p00["added"], "car"].to_numpy())
    s00 = st.summarize(p00, prices=prices, tickers=data.TICKERS, window=(0, 0), n_draws=8000)
    print(f"  added [0,0]: mean={a00['mean_pct']:+.2f}%  t={a00['t']:+.2f}  "
          f"placebo p={s00['added_placebo_p']:.3f}   added-dropped diff t={d00:+.2f}")

    print("\n# The TRADABLE version — enter 1 day after (lag=1), hold the window")
    for w, lab in [((0, 2), "[+1,+3]"), ((0, 4), "[+1,+5]")]:
        pl = st.car_panel(prices, events, window=w, lag=1)
        al = st.summarize_bucket(pl.loc[pl["added"], "car"].to_numpy())
        dl = st.welch_t(pl.loc[pl["added"], "car"].to_numpy(),
                        pl.loc[~pl["added"], "car"].to_numpy())
        nc = st.net_of_costs(al["mean_pct"] / 100)
        print(f"  lag1 {lab}: added mean={al['mean_pct']:+.2f}%  t={al['t']:+.2f}  "
              f"diff t={dl:+.2f}  net@10bps={nc['net_pct']:+.2f}%")

    print("\n# Robustness — vary the event window")
    for w in [(0, 0), (0, 2), (-1, 1), (0, 4)]:
        pw = st.car_panel(prices, events, window=w)
        sw = st.summarize(pw)
        print(f"  window {str(w):>8}: added mean={sw['added']['mean_pct']:+.2f}%  "
              f"t={sw['added']['t']:+.2f}  dropped={sw['dropped']['mean_pct']:+.2f}%  "
              f"diff={sw['diff_pct']:+.2f}pp  diff t={sw['diff_t']:+.2f}")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED added-bucket CAR edge and must NOT manufacture")
print("  significance from ~a dozen events per bucket when the true edge is 0.")
for edge in (0.0, 500.0):
    syn = data.synthetic_events(car_bps=edge, seed=751)
    ac = st.summarize_bucket(syn["added_car"])
    dc = st.summarize_bucket(syn["dropped_car"])
    dt = st.welch_t(syn["added_car"], syn["dropped_car"])
    diff = (np.mean(syn["added_car"]) - np.mean(syn["dropped_car"])) * 100
    print(f"  planted car_bps={edge:+5.0f}: added mean={ac['mean_pct']:+.2f}% "
          f"t={ac['t']:+.2f}  dropped mean={dc['mean_pct']:+.2f}%  "
          f"diff={diff:+.2f}pp  diff t={dt:+.2f}")
