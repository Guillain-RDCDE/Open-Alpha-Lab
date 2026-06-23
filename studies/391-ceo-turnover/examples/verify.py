"""Reproducible headline run for Study 391 — CEO-Turnover.

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

from ceo_turnover import data, strategy as st

print("# CEO-Turnover — short-window market-model abnormal returns around CEO changes")
if data.have_real():
    prices, events = data.load_real()
    print(f"event tickers cached : {sorted(c for c in prices.columns if c != 'SPY')}")
    print(f"events with price data: {len(events)} of {len(data.CEO_EVENTS)} "
          f"(table fingerprint {data.fingerprint(data.CEO_EVENTS)})")

    panel = st.car_panel(prices, events)          # canonical CAR[0, +2] window
    print(f"\n# Canonical event window CAR[0,+2] (market model, SPY benchmark)")
    print(panel.to_string(index=False,
                          formatters={"car": lambda v: f"{v*100:+.2f}%"}))

    s = st.summarize(panel, prices=prices, tickers=data.TICKERS, n_draws=8000)
    print("\n# Buckets (forced vs planned) — CAR[0,+2]")
    for k in ("forced", "planned", "all"):
        b = s[k]
        print(f"  {k:>8}: n={b['n']:>2}  mean={b['mean_pct']:+6.2f}%  "
              f"win={b['win']*100:>4.0f}%  t(vs 0)={b['t']:+5.2f}")
    print(f"  forced - planned: {s['diff_pct']:+.2f}pp   Welch t = {s['diff_t']:+.2f}")
    print(f"  forced placebo p (random non-event windows) = {s['forced_placebo_p']:.3f}")

    print("\n# The announcement-day repricing you CANNOT trade (window [0,0])")
    p00 = st.car_panel(prices, events, window=(0, 0))
    f00 = st.summarize_bucket(p00.loc[p00["forced"], "car"].to_numpy())
    d00 = st.welch_t(p00.loc[p00["forced"], "car"].to_numpy(),
                     p00.loc[~p00["forced"], "car"].to_numpy())
    s00 = st.summarize(p00, prices=prices, tickers=data.TICKERS, window=(0, 0), n_draws=8000)
    print(f"  forced [0,0]: mean={f00['mean_pct']:+.2f}%  t={f00['t']:+.2f}  "
          f"placebo p={s00['forced_placebo_p']:.3f}   forced-planned diff t={d00:+.2f}")

    print("\n# The TRADABLE version — enter 1 day after (lag=1), hold the window")
    for w, lab in [((0, 2), "[+1,+3]"), ((0, 4), "[+1,+5]")]:
        pl = st.car_panel(prices, events, window=w, lag=1)
        fl = st.summarize_bucket(pl.loc[pl["forced"], "car"].to_numpy())
        dl = st.welch_t(pl.loc[pl["forced"], "car"].to_numpy(),
                        pl.loc[~pl["forced"], "car"].to_numpy())
        nc = st.net_of_costs(fl["mean_pct"] / 100)
        print(f"  lag1 {lab}: forced mean={fl['mean_pct']:+.2f}%  t={fl['t']:+.2f}  "
              f"diff t={dl:+.2f}  net@10bps={nc['net_pct']:+.2f}%")

    print("\n# Robustness — vary the event window")
    for w in [(0, 0), (0, 2), (-1, 1), (0, 4)]:
        pw = st.car_panel(prices, events, window=w)
        sw = st.summarize(pw)
        print(f"  window {str(w):>8}: forced mean={sw['forced']['mean_pct']:+.2f}%  "
              f"t={sw['forced']['t']:+.2f}  diff={sw['diff_pct']:+.2f}pp  "
              f"diff t={sw['diff_t']:+.2f}")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED forced-bucket CAR edge and must NOT manufacture")
print("  significance from ~a dozen events per bucket when the true edge is 0.")
for edge in (0.0, 500.0):
    syn = data.synthetic_events(car_bps=edge, seed=391)
    fc = st.summarize_bucket(syn["forced_car"])
    pl = st.summarize_bucket(syn["planned_car"])
    dt = st.welch_t(syn["forced_car"], syn["planned_car"])
    diff = (np.mean(syn["forced_car"]) - np.mean(syn["planned_car"])) * 100
    print(f"  planted car_bps={edge:+5.0f}: forced mean={fc['mean_pct']:+.2f}% "
          f"t={fc['t']:+.2f}  planned mean={pl['mean_pct']:+.2f}%  "
          f"diff={diff:+.2f}pp  diff t={dt:+.2f}")
