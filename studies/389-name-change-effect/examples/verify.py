"""Reproducible headline run for Study 389 — Name-Change-Effect.

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

from name_change_effect import data, strategy as st

print("# Name-Change-Effect — ~25 real theme-chasing rebrands (.com / blockchain / AI)")
print(f"rebrand table  : {len(data.REBRANDS)} rows across 3 waves; "
      f"{len(data.DELISTED)} famous ones DELISTED (survivorship, named)")

if data.have_real():
    b = data.load_real()
    ev = st.collect_events(b, pop=5, fade=60)
    print(f"events priced  : {len(ev)} of {len(data.REBRANDS)} "
          f"(rest delisted / listed after the announce / window overran)")
    print("\n# Per-event abnormal (excess-of-SPY) returns: pop[+1..+5d], fade[+6..+65d]")
    print(f"  {'ticker':>6} {'wave':>11} {'pop%':>8} {'fade%':>9}")
    for _, r in ev.iterrows():
        print(f"  {r['ticker']:>6} {r['wave']:>11} {r['pop']*100:>7.1f}% {r['fade']*100:>8.1f}%")

    s = st.summarize(ev, b, pop=5, fade=60)
    print("\n# The two legs vs zero (Welch t) and vs a placebo null sized to the event count")
    print(f"  {'leg':>5} {'n':>3} {'mean':>8} {'win':>6} {'Welch_t':>8} {'placebo_p':>10}")
    for leg in ("pop", "fade"):
        d = s[leg]
        print(f"  {leg:>5} {s['n']:>3} {d['mean']*100:>7.2f}% {d['win']*100:>5.0f}% "
              f"{d['t']:>8.2f} {d['p_placebo']:>10.3f}")

    print("\n# By wave (mean abnormal pop / fade)")
    for w in ("dotcom", "blockchain", "ai"):
        sub = ev[ev["wave"] == w]
        print(f"  {w:>11}: n={len(sub):>2}  pop={sub['pop'].mean()*100:>6.2f}%  "
              f"fade={sub['fade'].mean()*100:>7.2f}%")

    print("\n# Buy-the-pop / short-the-fade book, net of small-cap costs (one-way bps x 4 legs)")
    c = st.net_of_costs(ev, cost_bps=30.0)
    print(f"  gross combo (pop - fade) = {c['gross_combo']*100:>6.2f}%   "
          f"net @30bps/leg = {c['net_combo']*100:>6.2f}%   (n={c['n']})")

    print("\n# Robustness — shift the event windows")
    print(f"  {'pop/fade':>9} {'n':>3} {'pop_mean':>9} {'pop_t':>6} {'fade_mean':>10} {'fade_t':>7}")
    for pop, fade in ((3, 40), (5, 60), (10, 120)):
        e2 = st.collect_events(b, pop=pop, fade=fade)
        s2 = st.summarize(e2, b, pop=pop, fade=fade, placebo=False)
        print(f"  {str(pop)+'/'+str(fade):>9} {s2['n']:>3} {s2['pop']['mean']*100:>8.2f}% "
              f"{s2['pop']['t']:>6.2f} {s2['fade']['mean']*100:>9.2f}% {s2['fade']['t']:>7.2f}")
else:
    print("(no _cache/rebrand_prices.csv — run data.fetch_prices() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the engine must NOT find a significant pop/fade with edge=0, and MUST light up")
print("  both legs with a large planted pop+give-back.")
for edge in (0.0, 0.40):
    syn = data.synthetic_rebrands(n_events=25, edge=edge, seed=389)
    ev_s = st.collect_events(syn, pop=5, fade=60)
    s = st.summarize(ev_s, syn, pop=5, fade=60, placebo=False)
    print(f"  planted edge={edge:.2f}: n={s['n']:>2}  "
          f"pop_mean={s['pop']['mean']*100:>6.2f}% pop_t={s['pop']['t']:>5.2f}  "
          f"fade_mean={s['fade']['mean']*100:>7.2f}% fade_t={s['fade']['t']:>5.2f}")
