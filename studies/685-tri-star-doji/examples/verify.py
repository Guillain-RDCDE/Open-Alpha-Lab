"""Reproducible headline run for Study 685 — Tri-Star Doji.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from tri_star_doji import data, strategy as st  # noqa: E402

print("# Tri-Star Doji — do three dojis in a row mark a major reversal?")
print(f"basket: {len(data.BASKET)} names (SPY + {len(data.BASKET) - 1} long-listed US "
      f"large-caps), as-of {data.AS_OF}")

if not data.have_real():
    print("(cache miss — fetching the basket once)")
    data.load_real(fetch=True)

panel = data.load_real()
sample = data.BASKET[0]
print(data_stamp(f"{sample} OHLCV (sample)", panel[sample][["close"]], asof=data.AS_OF))
print(f"panel fingerprint: {data.panel_fingerprint(panel)}")

years = max((b.index[-1] - b.index[0]).days / 365.25 for b in panel.values())
print(f"window: ~{years:.1f} years per name, {len(panel)} names, "
      f"{sum(len(b) for b in panel.values()):,} total bars")

print("\n# THE DETECTOR — three consecutive dojis (body <= 10% of range)")
res = st.run_experiment(panel)
print(f"  loose cut  (3 dojis in a row, no gap requirement): n = {res['n_tri_star']}")
print(f"  strict cut (island tri-star, middle doji gaps both sides): "
      f"n = {res['n_gapped']}  <- the literature-faithful reading, and it is genuinely rare")
print(f"  Bonferroni-corrected critical |t| for {len(res['horizons'])} horizons: "
      f"{res['bonferroni_crit']:.2f}")

print("\n# STRICT cut — the literature-faithful island tri-star (both gaps)")
for h in st.HORIZONS:
    s = res["strict_per_horizon"][h]["tri"]
    if not s["tested"]:
        print(f"  {h:>2}d: n={s['n']} -- TOO FEW TO TEST (min {st.MIN_N_FOR_TEST}); "
              f"raw mean {s['mean_bps']:+.1f} bps, win rate {s['win_rate']*100:.0f}% "
              f"(descriptive only, no t-stat computed)")
    else:
        print(f"  {h:>2}d: n={s['n']}  mean {s['mean_bps']:+.1f} bps  t={s['tstat']:+.2f}")

print("\n# LOOSE cut — three dojis in a row, no gap requirement (the honest power check)")
print(f"  {'H':>3} {'n':>4} {'tri_bps':>9} {'win%':>5} {'base_bps':>9} {'delta_bps':>10} "
      f"{'welch_t':>8} {'placebo_p':>10}")
for h in st.HORIZONS:
    ph = res["per_horizon"][h]
    s, b = ph["tri"], ph["base"]
    wt = ph["welch_t"]
    wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
    print(f"  {h:>3} {s['n']:>4} {s['mean_bps']:>9.1f} {s['win_rate']*100:>5.0f} "
          f"{b['mean_bps']:>9.1f} {ph['delta_bps']:>10.1f} {wt_s:>8} {ph['placebo_p']:>10.3f}")

crit = res["bonferroni_crit"]
survivors = [h for h in st.HORIZONS
             if res["per_horizon"][h]["welch_t"] is not None
             and abs(res["per_horizon"][h]["welch_t"]) >= crit]
print(f"  horizons clearing the Bonferroni bar (|welch t| >= {crit:.2f}): "
      f"{survivors if survivors else 'NONE'}")

print("\n# Costs — net of 2 bps round-trip + 1 bp borrow on the short leg (loose cut)")
for h in st.HORIZONS:
    s = res["per_horizon"][h]["tri"]
    print(f"  {h:>2}d: gross {s['mean_bps']:+.1f} bps -> net {s['net_bps']:+.1f} bps")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch delta detector must NOT fire on a null world (edge=0) and must recover")
print("  a planted tri-star reversal. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    d, _truth = data.synthetic_panel(edge=0.0, seed=685 + s_)
    r = st.synthetic_detect(d, horizon=20, seed=685 + s_)
    if r["welch_t"] is not None:
        null_ts.append(r["welch_t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (edge=0), {len(null_ts)} seeds: mean welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/{len(null_ts)} seeds")
for edge in (0.02, 0.04):
    d, truth = data.synthetic_panel(edge=edge, seed=685)
    r = st.synthetic_detect(d, horizon=20, seed=685)
    print(f"  planted edge={edge:.2f} (seed 685): n={r['n']}  tri {r['mean_bps']:+.1f} bps "
          f"vs base {r['base_bps']:+.1f} bps  delta {r['delta_bps']:+.1f} bps  "
          f"Welch t={r['welch_t']:+.2f}")
