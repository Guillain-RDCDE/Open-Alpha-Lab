"""Reproducible headline run for Study 746 — HQ-Relocation.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily closes under
``_cache/`` if present (the real-tape event study), sliced to a frozen **as-of**, and
always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hq_relocation import data, strategy as st

try:
    from quantlab.repro import as_of, fingerprint as qfp  # noqa: F401
    HAVE_QL = True
except Exception:
    HAVE_QL = False

AS_OF = "2026-06-30"

print("# HQ-Relocation — short-window market-model abnormal returns around HQ-move announcements")
print(f"as-of          : {AS_OF}   (headline sample sliced to on-or-before this date)")

if data.have_real():
    prices, events = data.load_real()
    prices = prices[prices.index <= AS_OF]
    print(f"event tickers  : {sorted(c for c in prices.columns if c != 'SPY')}")
    print(f"events priced  : {len(events)} of {len(data.HQ_MOVES)} "
          f"(table fingerprint {data.fingerprint(data.HQ_MOVES)})")

    panel = st.car_panel(prices, events)          # canonical CAR[0, +2] window
    print(f"\n# Canonical event window CAR[0,+2] (market model, SPY benchmark)")
    print(panel.to_string(index=False,
                          formatters={"car": lambda v: f"{v*100:+.2f}%"}))

    s = st.summarize(panel, prices=prices, tickers=data.TICKERS, n_draws=8000)
    print("\n# Buckets (tax/incentive vs other) — CAR[0,+2]")
    for k in ("tax", "other", "all"):
        b = s[k]
        print(f"  {k:>6}: n={b['n']:>2}  mean={b['mean_pct']:+6.2f}%  "
              f"win={b['win']*100:>4.0f}%  t(vs 0)={b['t']:+5.2f}")
    print(f"  tax - other: {s['diff_pct']:+.2f}pp   Welch t = {s['diff_t']:+.2f}")
    print(f"  all-events placebo p (random non-event windows) = {s['all_placebo_p']:.3f}")

    print("\n# The announcement-day repricing (window [0,0])")
    p00 = st.car_panel(prices, events, window=(0, 0))
    a00 = st.summarize_bucket(p00["car"].to_numpy())
    s00 = st.summarize(p00, prices=prices, tickers=data.TICKERS, window=(0, 0), n_draws=8000)
    print(f"  all [0,0]: mean={a00['mean_pct']:+.2f}%  t={a00['t']:+.2f}  "
          f"win={a00['win']*100:.0f}%  placebo p={s00['all_placebo_p']:.3f}")

    print("\n# The post-announcement DRIFT the 'signal' camp needs — enter +1 day, hold a quarter")
    for w, lab in [((1, 21), "[+1,+21]  ~1mo"), ((1, 63), "[+1,+63]  ~1qtr")]:
        dp = st.car_panel(prices, events, window=w)
        da = st.summarize_bucket(dp["car"].to_numpy())
        dt = st.summarize_bucket(dp.loc[dp["tax"], "car"].to_numpy())
        nc = st.net_of_costs(da["mean_pct"] / 100)
        print(f"  drift {lab}: all mean={da['mean_pct']:+.2f}%  t={da['t']:+.2f}  "
              f"tax mean={dt['mean_pct']:+.2f}%  net@10bps={nc['net_pct']:+.2f}%")

    print("\n# Robustness — vary the event window")
    for w in [(0, 0), (0, 2), (-1, 1), (0, 4)]:
        pw = st.car_panel(prices, events, window=w)
        sw = st.summarize(pw)
        print(f"  window {str(w):>8}: all mean={sw['all']['mean_pct']:+.2f}%  "
              f"t={sw['all']['t']:+.2f}  tax-other={sw['diff_pct']:+.2f}pp  "
              f"diff t={sw['diff_t']:+.2f}")
else:
    print("(no _cache — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED tax-bucket CAR edge and must NOT manufacture")
print("  significance from ~a dozen events per bucket when the true edge is 0.")
for edge in (0.0, 500.0):
    syn = data.synthetic_events(car_bps=edge, seed=746)
    fc = st.summarize_bucket(syn["tax_car"])
    pl = st.summarize_bucket(syn["other_car"])
    dt = st.welch_t(syn["tax_car"], syn["other_car"])
    diff = (np.mean(syn["tax_car"]) - np.mean(syn["other_car"])) * 100
    print(f"  planted car_bps={edge:+5.0f}: tax mean={fc['mean_pct']:+.2f}% "
          f"t={fc['t']:+.2f}  other mean={pl['mean_pct']:+.2f}%  "
          f"diff={diff:+.2f}pp  diff t={dt:+.2f}")
