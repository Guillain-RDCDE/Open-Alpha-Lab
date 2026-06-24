"""Reproducible headline run for Study 412 — Symmetrical Triangle.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket parquets under
``_cache/`` for the real-tape numbers (offline once cached) and always runs the
synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from symmetrical_triangle import data, strategy as st  # noqa: E402

NDRAWS = 20_000

print("# Symmetrical Triangle — mechanical detector on SPY + 29 large-caps (yfinance daily)")

if data.have_real():
    panel = data.load_real()
    allidx = [b.index for b in panel.values()]
    lo = min(i.min() for i in allidx)
    hi = max(i.max() for i in allidx)
    yrs = (hi - lo).days / 365.25
    nbars = sum(len(b) for b in panel.values())
    print(f"names          : {len(panel)}")
    print(f"window         : {lo.date()} -> {hi.date()}  ({yrs:.1f} years, {nbars:,} bars)")
    print(f"fingerprint    : {data.fingerprint(panel)}")

    res = st.run_experiment(panel, n_draws=NDRAWS)
    led = res["ledger"]
    print(f"\nconfirmed breakouts: {res['n_events']}  "
          f"(up={int((led.dir == 1).sum())}, down={int((led.dir == -1).sum())})")

    print("\n# Forward return after a CONFIRMED breakout vs random-day base rate "
          "(signed by direction, 1-day entry lag)")
    print(f"  {'H':>3} {'n':>4} {'win%':>5} {'sig%':>7} {'base%':>7} {'excess%':>8} "
          f"{'t_exc':>6} {'t_sig':>6} {'p_perm':>7} {'net%':>7}")
    for h in st.HORIZONS:
        s = res["per_horizon"][h]
        print(f"  {h:>3} {s['n']:>4} {s['win_rate']*100:>4.1f} "
              f"{s['mean_sig']*100:>+6.3f} {s['mean_base']*100:>+6.3f} "
              f"{s['excess']*100:>+7.3f} {s['t_excess']:>+6.2f} {s['t_sig']:>+6.2f} "
              f"{s['p_perm']:>7.3f} {s['net']*100:>+6.3f}")

    print("\n# Direction split — up-breakout vs down-breakout (the 'continuation' myth check)")
    print(f"  {'H':>3}  {'UP n':>5} {'mean%':>7} {'win%':>5} {'t':>6}   "
          f"{'DN n':>5} {'mean%':>7} {'win%':>5} {'t':>6}")
    for h in st.HORIZONS:
        sp = res["split"][h]
        u, d = sp["up"], sp["down"]
        print(f"  {h:>3}  {u['n']:>5} {u['mean']*100:>+6.3f} {u['win']*100:>4.1f} "
              f"{u['t']:>+6.2f}   {d['n']:>5} {d['mean']*100:>+6.3f} "
              f"{d['win']*100:>4.1f} {d['t']:>+6.2f}")
else:
    print("(no _cache parquets — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector + inference must recover a PLANTED continuation edge and must NOT")
print("  manufacture significance when the true edge is 0.")
for edge in (0.0, 0.20):
    panel, _ = data.synthetic_panel(edge=edge, seed=412, n_names=40, n_triangles=8)
    res = st.run_experiment(panel, n_draws=5_000)
    s = res["per_horizon"][20]
    print(f"  planted edge={edge:+.2f}: events={res['n_events']:>4}  "
          f"sig20={s['mean_sig']*100:>+6.2f}%  excess={s['excess']*100:>+6.2f}%  "
          f"t_excess={s['t_excess']:>+5.2f}  t_sig={s['t_sig']:>+5.2f}  "
          f"p={s['p_perm']:.3f}  win={s['win_rate']*100:.0f}%")
