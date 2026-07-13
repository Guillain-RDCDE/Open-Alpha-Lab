"""Reproducible headline run for Study 720 — Super-Bowl-Advertiser.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached advertiser prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from super_bowl_advertiser import data, strategy as st

print("# Super-Bowl-Advertiser — ~32 real LISTED Super Bowl advertisers (2015-2024)")
print(f"advertiser table : {len(data.ADVERTISERS)} advertiser-year events; "
      f"{len(data.DELISTED)} famous advertisers DELISTED / went private (survivorship, named)")
print(f"fingerprint      : {data.fingerprint()}")

if data.have_real():
    b = data.load_real()
    ev = st.collect_events(b, drift=5, hold=20)
    print(f"events priced    : {len(ev)} of {len(data.ADVERTISERS)} "
          f"(rest listed after the game / went private / window overran)")
    print("\n# Per-event abnormal (excess-of-SPY) returns: monday reaction, drift[+1..+5d], hold[+6..+25d]")
    print(f"  {'ticker':>6} {'year':>5} {'monday%':>8} {'drift%':>8} {'hold%':>8}")
    for _, r in ev.iterrows():
        print(f"  {r['ticker']:>6} {r['year']:>5} {r['monday']*100:>7.1f}% "
              f"{r['drift']*100:>7.1f}% {r['hold']*100:>7.1f}%")

    s = st.summarize(ev, b, drift=5, hold=20)
    print("\n# The legs vs zero (Welch t) and vs a placebo null sized to the event count")
    print(f"  {'leg':>7} {'n':>3} {'mean':>8} {'win':>6} {'Welch_t':>8} {'placebo_p':>10}")
    for leg in ("monday", "drift", "hold"):
        d = s[leg]
        pp = d.get("p_placebo", float("nan"))
        print(f"  {leg:>7} {s['n']:>3} {d['mean']*100:>7.2f}% {d['win']*100:>5.0f}% "
              f"{d['t']:>8.2f} {pp:>10.3f}")

    print("\n# By year (mean abnormal drift / hold)")
    for y in sorted(ev["year"].unique()):
        sub = ev[ev["year"] == y]
        print(f"  {y}: n={len(sub):>2}  drift={sub['drift'].mean()*100:>6.2f}%  "
              f"hold={sub['hold'].mean()*100:>7.2f}%")

    print("\n# Long-the-advertisers ad-calendar book, net of large-cap costs (one-way bps x 2 legs)")
    c = st.net_of_costs(ev, cost_bps=10.0)
    print(f"  buy-the-drift [+1..+5d]  gross = {c['gross_drift']*100:>6.2f}%   "
          f"net @10bps/leg = {c['net_drift']*100:>6.2f}%   (n={c['n']})")
    print(f"  hold-through  [+1..+25d] gross = {c['gross_hold']*100:>6.2f}%   "
          f"net @10bps/leg = {c['net_hold']*100:>6.2f}%")

    print("\n# Robustness — shift the event windows")
    print(f"  {'drift/hold':>10} {'n':>3} {'drift_mean':>11} {'drift_t':>8} {'hold_mean':>10} {'hold_t':>7}")
    for drift, hold in ((3, 10), (5, 20), (10, 40)):
        e2 = st.collect_events(b, drift=drift, hold=hold)
        s2 = st.summarize(e2, b, drift=drift, hold=hold, placebo=False)
        print(f"  {str(drift)+'/'+str(hold):>10} {s2['n']:>3} {s2['drift']['mean']*100:>10.2f}% "
              f"{s2['drift']['t']:>8.2f} {s2['hold']['mean']*100:>9.2f}% {s2['hold']['t']:>7.2f}")
else:
    print("(no _cache/superbowl_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must NOT find a significant drift with edge=0, and MUST light up the")
print("  drift leg with a large planted post-game drift.")
for edge in (0.0, 0.10):
    syn = data.synthetic_ads(n_events=32, edge=edge, seed=726)
    ev_s = st.collect_events(syn, drift=5, hold=20)
    s = st.summarize(ev_s, syn, drift=5, hold=20, placebo=False)
    print(f"  planted edge={edge:.2f}: n={s['n']:>2}  "
          f"drift_mean={s['drift']['mean']*100:>6.2f}% drift_t={s['drift']['t']:>6.2f}  "
          f"hold_mean={s['hold']['mean']*100:>6.2f}% hold_t={s['hold']['t']:>5.2f}")
