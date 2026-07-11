"""Reproducible headline run for Study 692 — Breakaway Candles.

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

from breakaway_candles import data, strategy as st  # noqa: E402

print("# Breakaway Candles — does the 5-candle gap/run/reversal figure mark a real "
      "trend reversal?")
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

print("\n# THE DETECTOR — gap + 2-day run + long reversal candle closing back through "
      "the window")
res = st.combined_experiment(panel)
print(f"  combined (bullish + bearish) loose cut: n = {res['n_loose']} "
      f"(bullish {res['n_bullish']}, bearish {res['n_bearish']})")
print(f"  strict cut (bigger gap, long candles 1 & 5, full gap fill): n = {res['n_strict']}"
      "  <- the literature-closer reading, and it is rarer still")
print(f"  Bonferroni-corrected critical |t| for {len(res['horizons'])} horizons: "
      f"{res['bonferroni_crit']:.2f}")

print("\n# HEADLINE — combined loose cut vs the trend-matched base rate")
print(f"  {'H':>3} {'n':>5} {'reversal_bps':>12} {'win%':>5} {'base_bps':>9} "
      f"{'delta_bps':>10} {'welch_t':>8} {'placebo_p':>10} {'net_bps':>8}")
for h in st.HORIZONS:
    ph = res["per_horizon"][h]
    s, b = ph["ladder"], ph["base"]
    wt = ph["welch_t"]
    wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
    print(f"  {h:>3} {s['n']:>5} {s['mean_bps']:>12.1f} {s['win_rate']*100:>5.1f} "
          f"{b['mean_bps']:>9.1f} {ph['delta_bps']:>10.1f} {wt_s:>8} {ph['placebo_p']:>10.3f} "
          f"{s['net_bps']:>8.1f}")

crit = res["bonferroni_crit"]
survivors = [h for h in st.HORIZONS
            if res["per_horizon"][h]["welch_t"] is not None
            and abs(res["per_horizon"][h]["welch_t"]) >= crit]
print(f"\n  horizons clearing the Bonferroni bar (|welch t| >= {crit:.2f}): "
      f"{survivors if survivors else 'NONE'}")

print("\n# Per-side breakdown — the desk's own symmetry myth-check")
side_res = {}
for side in ("bullish", "bearish"):
    r = st.run_experiment(panel, side)
    side_res[side] = r
    print(f"  {side} loose cut: n = {r['n_loose']}  "
          f"{'H':>3} {'n':>4} {'mean_bps':>9} {'base_bps':>9} {'delta_bps':>9} "
          f"{'welch_t':>8} {'placebo_p':>9}")
    for h in st.HORIZONS:
        ph = r["per_horizon"][h]
        s, b = ph["ladder"], ph["base"]
        wt = ph["welch_t"]
        wt_s = f"{wt:+.2f}" if wt is not None else "n/a"
        print(f"        {h:>3} {s['n']:>4} {s['mean_bps']:>9.1f} {b['mean_bps']:>9.1f} "
              f"{ph['delta_bps']:>9.1f} {wt_s:>8} {ph['placebo_p']:>9.3f}")

print("\n# Best / worst confirmed bullish breakaways (20-day forward, bps) — is it the "
      "candle, or the calendar?")
ev = st.pool_events(panel, "bullish")
recs = []
for _, row in ev.iterrows():
    tkr = row["ticker"]
    bars = panel[tkr]
    pos = int(row["pos"])
    d0 = bars.index[pos - 4].date()
    d1 = bars.index[pos].date()
    recs.append((tkr, str(d0), str(d1), row["ret_20"] * 1e4))
recs.sort(key=lambda r: r[3])
for tkr, d0, d1, r20 in recs[:2] + recs[-3:]:
    print(f"  {tkr}: block {d0} -> {d1}   20d {r20:+.1f} bps")

print("\n# Synthetic positive control — deterministic, no network")
print("  the combined Welch delta detector must NOT fire on a null world (edge=0) and")
print("  must recover a planted breakaway drift. Null checked over 20 seeds (never a")
print("  single stream).")
null_ts = []
for s_ in range(20):
    d, _truth = data.synthetic_panel(edge=0.0, seed=692 + s_)
    r = st.synthetic_detect_combined(d, horizon=20, seed=692 + s_)
    if r["welch_t"] is not None:
        null_ts.append(r["welch_t"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (edge=0), {len(null_ts)} seeds fired a testable sample: "
      f"mean welch t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/{len(null_ts)} seeds")
for edge in (0.02, 0.04):
    d, truth = data.synthetic_panel(edge=edge, seed=692)
    r = st.synthetic_detect_combined(d, horizon=20, seed=692)
    print(f"  planted edge={edge:.2f} (seed 692): n={r['n']}  reversal {r['mean_bps']:+.1f} "
          f"bps vs base {r['base_bps']:+.1f} bps  delta {r['delta_bps']:+.1f} bps  "
          f"Welch t={r['welch_t']:+.2f}")
