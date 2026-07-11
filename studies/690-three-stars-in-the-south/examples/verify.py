"""Reproducible headline run for Study 690 — Three Stars in the South.

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

from three_stars_in_the_south import data, strategy as st  # noqa: E402

print("# Three Stars in the South — do three shrinking black candles with rising lows "
      "end a downtrend?")
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
tot = sum(len(b) for b in panel.values())
print(f"window: ~{years:.1f} years per name, {len(panel)} names, {tot:,} total bars")

print("\n# THE DETECTOR — three shrinking-range bearish candles with rising lows, in a "
      "downtrend")
res = st.run_experiment(panel)
print(f"  loose cut  (3 shrinking bearish candles + rising lows + downtrend context): "
      f"n = {res['n_loose']}")
print(f"  strict cut (+ hammer first star, no gap-down second star, marubozu third "
      f"star): n = {res['n_strict']}  <- the literature-closer reading, and it is "
      "genuinely rare")
print(f"  Bonferroni-corrected critical |t| for {len(res['horizons'])} horizons: "
      f"{res['bonferroni_crit']:.2f}")

print("\n# LOOSE cut — vs the downtrend-matched base rate")
print(f"  {'H':>3} {'n':>5} {'star_bps':>9} {'win%':>5} {'base_bps':>9} {'delta_bps':>10} "
      f"{'welch_t':>8} {'placebo_p':>10} {'net_bps':>8}")
for h in st.HORIZONS:
    ph = res["per_horizon"][h]
    s, b = ph["star"], ph["base"]
    wt = ph["welch_t"]
    wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
    print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win_rate']*100:>5.1f} "
          f"{b['mean_bps']:>9.1f} {ph['delta_bps']:>10.1f} {wt_s:>8} {ph['placebo_p']:>10.3f} "
          f"{s['net_bps']:>8.1f}")

print("\n# STRICT cut — the literature-closer three stars in the south")
print(f"  {'H':>3} {'n':>5} {'star_bps':>9} {'win%':>5} {'delta_bps':>10} "
      f"{'welch_t':>8} {'placebo_p':>10} {'net_bps':>8}")
for h in st.HORIZONS:
    sp = res["strict_per_horizon"][h]
    s = sp["star"]
    wt = sp["welch_t"]
    wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
    print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>9.1f} {s['win_rate']*100:>5.1f} "
          f"{sp['delta_bps']:>10.1f} {wt_s:>8} {sp['placebo_p']:>10.3f} {s['net_bps']:>8.1f}")

crit = res["bonferroni_crit"]
loose_survivors = [h for h in st.HORIZONS
                   if res["per_horizon"][h]["welch_t"] is not None
                   and abs(res["per_horizon"][h]["welch_t"]) >= crit]
strict_survivors = [h for h in st.HORIZONS
                    if res["strict_per_horizon"][h]["welch_t"] is not None
                    and abs(res["strict_per_horizon"][h]["welch_t"]) >= crit]
print(f"\n  horizons clearing the Bonferroni bar (|welch t| >= {crit:.2f}): "
      f"loose {loose_survivors if loose_survivors else 'NONE'}  "
      f"strict {strict_survivors if strict_survivors else 'NONE'}")

print("\n# Strict cut — best and worst 20-day outcomes")
ev = st.pool_events(panel)
strict_ev = ev[ev["strict"]].copy()
recs = []
for _, row in strict_ev.iterrows():
    tkr = row["ticker"]
    bars = panel[tkr]
    pos = int(row["pos"])
    d0 = bars.index[pos - 2].date()
    d1 = bars.index[pos].date()
    recs.append((tkr, str(d0), str(d1), row["ret_20"] * 1e4))
recs.sort(key=lambda r: r[3])
for tkr, d0, d1, r20 in recs[:2] + recs[-3:]:
    print(f"  {tkr}: block {d0} -> {d1}   20d {r20:+.1f} bps")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch delta detector must NOT fire on a null world (edge=0) and must recover")
print("  a planted three-stars bounce. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    d, _truth = data.synthetic_panel(edge=0.0, seed=690 + s_)
    r = st.synthetic_detect(d, horizon=20, seed=690 + s_,
                            active_masks=_truth["active_masks"])
    if r["welch_t"] is not None:
        null_ts.append(r["welch_t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (edge=0), {len(null_ts)} seeds: mean welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/{len(null_ts)} seeds")
for edge in (0.02, 0.04):
    d, truth = data.synthetic_panel(edge=edge, seed=690)
    r = st.synthetic_detect(d, horizon=20, seed=690, active_masks=truth["active_masks"])
    print(f"  planted edge={edge:.2f} (seed 690): n={r['n']}  star {r['mean_bps']:+.1f} bps "
          f"vs base {r['base_bps']:+.1f} bps  delta {r['delta_bps']:+.1f} bps  "
          f"Welch t={r['welch_t']:+.2f}")
