"""Reproducible headline run for Study 722 — Logo-Rebrand.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached rebrand prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from logo_rebrand import data, strategy as st

print("# Logo-Rebrand — ~26 real corporate rebrands / logo changes 2010-2025")
print(f"rebrand table  : {len(data.REBRANDS)} rows (name / identity / logo); "
      f"{len(data.DELISTED)} famous ones DELISTED / went private (survivorship, named)")

try:
    from quantlab import repro
    if data.have_real():
        p = repro.as_of(data.load_prices(), "2026-06-30")
        print(f"data stamp     : {len(p)} rows {p.index.min().date()}->{p.index.max().date()} "
              f"as-of 2026-06-30  fp={repro.fingerprint(p)}")
except Exception as e:  # pragma: no cover - repro is a convenience only
    print(f"(quantlab.repro unavailable: {e})")

if data.have_real():
    b = data.load_real()
    ev = st.collect_events(b, announce=5, drift=120)
    print(f"events priced  : {len(ev)} of {len(data.REBRANDS)} "
          f"(rest delisted / listed after the announce / window overran)")
    print("\n# Per-event abnormal (excess-of-SPY) returns: announce[+1..+5d], drift[+6..+126d]")
    print(f"  {'ticker':>6} {'kind':>9} {'announce%':>10} {'drift%':>9}")
    for _, r in ev.iterrows():
        print(f"  {r['ticker']:>6} {r['kind']:>9} {r['announce']*100:>9.1f}% {r['drift']*100:>8.1f}%")

    s = st.summarize(ev, b, announce=5, drift=120)
    print("\n# The two legs vs zero (Welch t) and vs a placebo null sized to the event count")
    print(f"  {'leg':>9} {'n':>3} {'mean':>8} {'win':>6} {'Welch_t':>8} {'placebo_p':>10}")
    for leg in ("announce", "drift"):
        d = s[leg]
        print(f"  {leg:>9} {s['n']:>3} {d['mean']*100:>7.2f}% {d['win']*100:>5.0f}% "
              f"{d['t']:>8.2f} {d['p_placebo']:>10.3f}")

    print("\n# By rebrand kind (mean abnormal announce / drift)")
    for k in ("name", "identity", "logo"):
        sub = ev[ev["kind"] == k]
        print(f"  {k:>9}: n={len(sub):>2}  announce={sub['announce'].mean()*100:>6.2f}%  "
              f"drift={sub['drift'].mean()*100:>7.2f}%")

    print("\n# Fragility of the week-one pop — drop the largest-|announce| events")
    a = ev["announce"].values
    order = np.argsort(-np.abs(a))
    for drop in (0, 1, 2, 3):
        keep = np.ones(len(a), bool)
        keep[order[:drop]] = False
        print(f"  drop {drop}: n={keep.sum():>2}  announce={a[keep].mean()*100:>6.2f}%  "
              f"t={st.welch_t(a[keep]):>6.2f}")
    print(f"  biggest-|announce| names: {[ev['ticker'].values[i] for i in order[:3]]}")

    print("\n# Buy-the-rebrand-and-hold book, net of large-cap costs (one-way bps x 2 crossings)")
    c = st.net_of_costs(ev, cost_bps=10.0)
    print(f"  gross hold (announce + drift) = {c['gross_hold']*100:>6.2f}%   "
          f"net @10bps x2 = {c['net_hold']*100:>6.2f}%   (n={c['n']})")

    print("\n# Robustness — shift the event windows")
    print(f"  {'ann/drift':>9} {'n':>3} {'ann_mean':>9} {'ann_t':>6} {'drift_mean':>11} {'drift_t':>8}")
    for ann, drift in ((3, 60), (5, 120), (5, 252), (10, 120), (1, 120)):
        e2 = st.collect_events(b, announce=ann, drift=drift)
        s2 = st.summarize(e2, b, announce=ann, drift=drift, placebo=False)
        print(f"  {str(ann)+'/'+str(drift):>9} {s2['n']:>3} {s2['announce']['mean']*100:>8.2f}% "
              f"{s2['announce']['t']:>6.2f} {s2['drift']['mean']*100:>10.2f}% {s2['drift']['t']:>8.2f}")
else:
    print("(no _cache/rebrand_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must NOT find a significant drift with edge=0, and MUST light up the")
print("  drift leg with a large planted renewal drift.")
for edge in (0.0, 0.30):
    syn = data.synthetic_rebrands(n_events=26, edge=edge, seed=722)
    ev_s = st.collect_events(syn, announce=5, drift=120)
    s = st.summarize(ev_s, syn, announce=5, drift=120, placebo=False)
    print(f"  planted edge={edge:.2f}: n={s['n']:>2}  "
          f"announce_mean={s['announce']['mean']*100:>6.2f}% announce_t={s['announce']['t']:>5.2f}  "
          f"drift_mean={s['drift']['mean']*100:>7.2f}% drift_t={s['drift']['t']:>5.2f}")
