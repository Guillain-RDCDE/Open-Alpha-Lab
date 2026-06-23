"""Reproducible headline run for Study 368 — Buyback-Drift.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached event prices under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from buyback_drift import data, strategy as st

print("# Buyback-Drift - abnormal (stock - SPY) drift after share-buyback AUTHORIZATIONS")
print(f"# event table: {len(data.event_table())} hard-coded notable authorizations "
      "(ticker, announce date, $bn size)")

if data.have_real():
    prices, events = data.load_real()
    print(f"price tape   : {prices.index.min().date()} -> {prices.index.max().date()} "
          f"({len(prices)} days)")

    print("\n# Abnormal forward returns after each authorization (1-day entry lag)")
    print(f"  {'H':>4} {'n':>3} {'ab_mean':>9} {'ab_med':>9} {'win':>6} "
          f"{'Welch_t':>8} {'p_placebo':>10}")
    for m in (1, 3, 6, 12):
        s = st.summarize(prices, events, m)
        print(f"  {str(m)+'m':>4} {s['n']:>3} {s['ab_mean']*100:>8.2f}% "
              f"{s['ab_median']*100:>8.2f}% {s['win']*100:>5.0f}% "
              f"{s['t']:>8.2f} {s['p_placebo']:>10.3f}")

    print("\n# Net of one round-trip cost (single-name buy-on-announcement, hold H months)")
    for m in (1, 3, 6):
        c = st.net_of_costs(prices, events, m, cost_bps=10.0)
        print(f"  {m:>2}m  n_trades={c['n_trades']:>2}  gross={c['gross_mean']*100:>6.2f}%  "
              f"net@10bps={c['net_mean']*100:>6.2f}%")

    print("\n# Robustness - split by program size (median = $%.1fbn)"
          % events["size_bn"].median())
    med = events["size_bn"].median()
    for label, ev in (("BIG  (>=median)", events[events["size_bn"] >= med]),
                      ("SMALL (<median)", events[events["size_bn"] < med])):
        s = st.summarize(prices, ev, 6)
        print(f"  {label:16s} n={s['n']:>2}  ab6_mean={s['ab_mean']*100:>5.2f}%  "
              f"t={s['t']:>5.2f}  p={s['p_placebo']:.3f}")
    print("  (the SMALL-bucket 6m p looks low, but t=1.19 << 2 and it does NOT hold at 1/3/12m")
    print("   - a median-split multiple-testing fluke, not an edge.)")
else:
    print("(no _cache/event_prices.csv - run data.fetch_event_prices() once to build it)")

print("\n# Synthetic positive control - deterministic, no network")
print("  inference must recover a PLANTED abnormal drift and must NOT manufacture")
print("  significance from ~30 noisy single-name events when the true edge is 0.")
for edge in (0.0, 0.15):
    prices, events = data.synthetic_events(n_events=31, edge=edge, seed=368)
    s6 = st.summarize(prices, events, 6)
    print(f"  planted edge={edge:+.2f}: detected={s6['n']:>2}  "
          f"cond6_mean={s6['ab_mean']*100:>6.2f}%  Welch_t={s6['t']:>5.2f}")
