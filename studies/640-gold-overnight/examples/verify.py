"""Reproducible headline run for Study 640 — Gold-Overnight ("Gold Trades in Its Sleep").

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached GLD/IAU/SPY open-close tape
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from gold_overnight import data, strategy as st
from quantlab import repro

AS_OF = "2026-06-30"       # last complete month before the run date
NDRAWS = 20_000

print('# Gold-Overnight — "Gold Trades in Its Sleep" (GLD + IAU confirm, SPY placebo)')
if data.have_real():
    wide = repro.as_of(data.load_real(), AS_OF)
    print(repro.data_stamp("go_ohlc (GLD/IAU/SPY adj open+close)", wide, asof=AS_OF))
    gld = data.session_returns(wide, "GLD")
    iau = data.session_returns(wide, "IAU")
    spy = data.session_returns(wide, "SPY")

    print("\n# Session split — overnight (close->open) vs intraday (open->close), bps/day")
    stats = {}
    for name, r in (("GLD", gld), ("IAU", iau), ("SPY", spy)):
        s = st.split_stats(r, n_draws=NDRAWS)
        stats[name] = s
        print(f"  {name}: {s['start']} -> {s['end']}  n={s['n_days']:,} days")
        print(f"       overnight {s['on_bps']:+.2f} bps/d (HAC t={s['t_on']:+.2f})  "
              f"intraday {s['id_bps']:+.2f} bps/d (HAC t={s['t_id']:+.2f})")
        print(f"       gap (on-id) {s['gap_bps']:+.2f} bps/d  HAC t={s['t_gap']:+.2f}  "
              f"Welch t={s['welch_gap']:+.2f}  sign-flip p={s['p_signflip']:.4f}")
        print(f"       cumulative: overnight-only {s['on_total_x']:.2f}x  "
              f"intraday-only {s['id_total_x']:.2f}x  "
              f"(ann: on {s['on_ann_pct']:+.2f}% / id {s['id_ann_pct']:+.2f}% / "
              f"close-to-close {s['cc_ann_pct']:+.2f}%)")

    print("\n# Sub-period decay — pre vs post the 2015-03-20 LBMA gold-fix reform "
          "(externally-dated break)")
    for name, r in (("GLD", gld), ("IAU", iau)):
        sp = st.subperiod_stats(r, data.FIX_REFORM)
        a, b = sp["pre"], sp["post"]
        print(f"  {name} pre  ({a['start']} -> {a['end']}, n={a['n_days']:,}): "
              f"on {a['on_bps']:+.2f} id {a['id_bps']:+.2f} gap {a['gap_bps']:+.2f} bps/d "
              f"(HAC t={a['t_gap']:+.2f})")
        print(f"  {name} post ({b['start']} -> {b['end']}, n={b['n_days']:,}): "
              f"on {b['on_bps']:+.2f} id {b['id_bps']:+.2f} gap {b['gap_bps']:+.2f} bps/d "
              f"(HAC t={b['t_gap']:+.2f})")
        print(f"  {name} change-in-gap Welch t = {sp['welch_change']:+.2f} "
              f"(positive would mean the gap SHRANK after the reform)")

    print("\n# Equity placebo — gold's night/day gap vs SPY's on the identical calendar")
    vp = st.vs_placebo(gld, spy)
    print(f"  common days n={vp['n_days']:,}: gold gap {vp['gold_gap_bps']:+.2f} bps/d "
          f"(HAC t={vp['t_gold']:+.2f})  SPY gap {vp['spy_gap_bps']:+.2f} bps/d "
          f"(HAC t={vp['t_spy']:+.2f})")
    print(f"  gold-vs-SPY difference Welch t = {vp['welch_gold_vs_spy']:+.2f}")

    print("\n# Harvest test (third axis) — buy MOC, sell next MOO, flat all day; "
          "2 one-way trades/day")
    for label, r in (("full period", gld),
                     ("pre-reform", gld[gld.index < pd.Timestamp(data.FIX_REFORM)]),
                     ("post-reform", gld[gld.index >= pd.Timestamp(data.FIX_REFORM)])):
        print(f"  GLD {label}:")
        for h in st.harvest_table(r):
            print(f"    cost={h['cost_bps']:>3.1f} bps/leg: gross {h['gross_bps_day']:+.2f} "
                  f"-> net {h['net_bps_day']:+.2f} bps/d  net ann {h['net_ann_pct']:+.2f}%  "
                  f"vs buy&hold {h['bh_ann_pct']:+.2f}%  (HAC t net = {h['t_net']:+.2f})")
else:
    print("(no _cache/go_ohlc.csv — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic control — deterministic, no network (machinery proof only)")
print("  the night/day split must NOT manufacture a gap on a null world and must recover")
print("  a planted overnight drift.")
for edge in (0.0, 0.0004):
    w = data.synthetic_world(night_edge=edge, seed=640)
    r = data.session_returns(w, "SYN")
    s = st.split_stats(r, n_draws=4_000)
    print(f"  planted night edge={edge*1e4:+.1f} bps/d: gap={s['gap_bps']:+.2f} bps/d  "
          f"HAC t={s['t_gap']:+.2f}  Welch t={s['welch_gap']:+.2f}  "
          f"sign-flip p={s['p_signflip']:.3f}")
