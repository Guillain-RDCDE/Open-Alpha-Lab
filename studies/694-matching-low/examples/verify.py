"""Reproducible headline run for Study 694 — Matching Low.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket OHLCV under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from matching_low import data, strategy as st  # noqa: E402

NDRAWS = 5_000
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-10)

print("# Matching Low — does a repeated close on two down candles mark support?")
if not data.have_real():
    print("(cache miss — fetching the 30-name basket once)")
    for t in data.BASKET:
        data.load_one(t, allow_fetch=True)

panel = data.load_real(allow_fetch=False, asof=AS_OF)
fp = data.fingerprint(panel)
mins = min(b.index.min() for b in panel.values())
maxs = max(b.index.max() for b in panel.values())
span = (maxs - mins).days / 365.25
tot = sum(len(b) for b in panel.values())
closes = pd.DataFrame({tk: b["close"] for tk, b in panel.items()})
print(data_stamp("basket daily closes (30 names)", closes, asof=AS_OF))
print(f"names          : {len(panel)}  (basket fingerprint {fp})")
print(f"window         : {mins.date()} -> {maxs.date()}  as-of {AS_OF}  "
      f"({span:.1f} years, {tot:,} bars)")

print("\n# Forward return after a matching low (enter next open, 1-day lag)")
print("  (base = the basket's own unconditional forward-return pool -- what an always-")
print("   long trader earns on the same names/window; the pattern is long-only)")
print(f"  {'H':>3} {'n':>5} {'mean%':>7} {'base%':>7} {'hit%':>5} {'basehit%':>8} "
      f"{'t_hac':>6} {'t_one':>6} {'t_welch':>7} {'p_plac':>7} {'net%':>7}")
headline = {}
for h in st.HORIZONS:
    s = st.summarize(panel, h, n_draws=NDRAWS)
    headline[h] = s
    print(f"  {h:>3} {s['n_events']:>5} {s['mean']*100:>6.3f} {s['base_mean']*100:>6.3f} "
          f"{s['hit']*100:>4.0f} {s['base_hit']*100:>7.0f} {s['t_hac']:>6.2f} "
          f"{s['t_one']:>6.2f} {s['t_welch']:>7.2f} {s['p_placebo']:>7.3f} "
          f"{s['net']*100:>6.3f}")
    print(f"       hit-rate Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%]")

print("\n# The long timer — cost sweep, 5 / 10 bps one-way (round trip), no borrow "
      "(long only)")
for h in st.HORIZONS:
    ev = st.collect_events(panel, h)
    g = headline[h]["mean"]
    n5 = float(st.event_net_returns(ev, cost_bps=5.0).mean())
    n10 = float(st.event_net_returns(ev, cost_bps=10.0).mean())
    print(f"  H={h:>2}: gross {g*100:+.3f}%  ->  net@5bps {n5*100:+.3f}%   "
          f"net@10bps {n10*100:+.3f}%")

print("\n# Bonferroni across the basket — per-ticker Welch t (vs that ticker's own")
print("  unconditional pool) at H=5")
pt = st.per_ticker_stats(panel, 5)
k = len(pt)
zc = st.bonferroni_z(k)
n_survive = int((pt["t_welch"].abs() >= zc).sum()) if k else 0
print(f"  {k} tickers with >= 6 events; Bonferroni-corrected |t| bar = {zc:.2f} "
      f"(uncorrected 1.96)")
print(f"  tickers clearing the corrected bar: {n_survive}/{k}")
if k:
    best = pt.reindex(pt["t_welch"].abs().sort_values(ascending=False).index).head(3)
    for _, row in best.iterrows():
        print(f"    {row['ticker']:>5}: n={int(row['n']):>3}  mean={row['mean']*100:>6.3f}%  "
              f"t_welch={row['t_welch']:>6.2f}")

print("\n# Myth check #1 — the STRICT close match "
      f"(|C1-C0|/|C0| <= {st.STRICT_TOL*100:.2f}%, vs the default {st.DEFAULT_TOL*100:.2f}%)")
for h in st.HORIZONS:
    s = st.summarize(panel, h, n_draws=NDRAWS, tol=st.STRICT_TOL)
    print(f"  H={h:>2}: n={s['n_events']:>4}  mean={s['mean']*100:>6.3f}%  "
          f"t_welch={s['t_welch']:>6.2f}  t_hac={s['t_hac']:>6.2f}  "
          f"p={s['p_placebo']:.3f}  net={s['net']*100:>6.3f}%")

print("\n# Myth check #2 — only matching lows AFTER a genuine prior downtrend "
      "(a real support test, not a random pair of red days)")
for h in st.HORIZONS:
    s = st.summarize(panel, h, n_draws=NDRAWS, require_downtrend=True)
    print(f"  H={h:>2}: n={s['n_events']:>4}  mean={s['mean']*100:>6.3f}%  "
          f"t_welch={s['t_welch']:>6.2f}  t_hac={s['t_hac']:>6.2f}  "
          f"p={s['p_placebo']:.3f}  net={s['net']*100:>6.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector + inference must recover a PLANTED post-pattern reversal and must NOT")
print("  manufacture significance when the true edge is 0. The certifying statistic is")
print("  Welch t vs the panel's own unconditional base.")
import numpy as np  # noqa: E402

null_ts = []
for s_ in range(20):
    px, _ = data.synthetic_panel(edge=0.0, seed=694 + s_)
    null_ts.append(st.summarize(px, 5, n_draws=1_000, placebo=False)["t_welch"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
for edge in (0.0, 0.0022, 0.0035):
    px, truth = data.synthetic_panel(edge=edge, seed=694)
    s = st.summarize(px, 5, n_draws=4_000)
    print(f"  planted edge={edge:+.3f}/day (seed 694): planted_days={truth['n_planted_days']:>4}  "
          f"n={s['n_events']:>4}  mean={s['mean']*100:>6.3f}%  t_welch={s['t_welch']:>6.2f}  "
          f"t_hac={s['t_hac']:>6.2f}  hit={s['hit']*100:.0f}%")
