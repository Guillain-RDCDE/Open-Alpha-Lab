"""Reproducible headline run for Study 781 — Quad-Witching-Hangover.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY tape under ``_cache/``
(fetching once on a cache miss), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from quad_witching_hangover import data as dt, strategy as st  # noqa: E402

print("# Quad-Witching-Hangover — does the week AFTER quad-witching underperform?")

print(f"calendar: {len(dt.EVENTS)} quad-witching Fridays {dt.EVENTS[0][1]}->{dt.EVENTS[-1][1]}, "
      f"hardcoded (third Friday of Mar/Jun/Sep/Dec, CBOE/CME expiration calendar)")

if not dt.have_real():
    print("(cache miss — fetching SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Quad-witching panel (SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} quad-witching Fridays have SPY "
      f"coverage around the anchor")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# THE HANGOVER — SPY raw forward return, K sessions AFTER the quad-witching close")
for label, col in (("1-week hangover (k=5)", "post_s"), ("2-week give-back (k=10)", "post_l")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<26s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"up {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE RUN-IN (contrast) — SPY return, K sessions BEFORE the quad-witching close")
s = st.one_sample_t(inc["pre"].values)
hr = st.hit_rate(inc["pre"].values)
print(f"  {'1-week run-in (k=5)':<26s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
      f"up {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
      f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event, left tail = underperform)")
for label, col, k in (("1wk hangover", "post_s", 5), ("2wk give-back", "post_l", 10)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail="left")
    print(f"  {label:<14s} (left-tail): observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — 1-week hangover t-stat")
x = inc["post_s"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — the hangover net of costs (calendar-known entry, zero look-ahead)")
for base, label in (("post_s", "1-week hangover"), ("post_l", "2-week give-back")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<16s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(dip=0.0, seed=793 + s, k=5)["t"] for s in range(20)])
print(f"  null (dip=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for dip in (0.006, 0.012):
    planted = st.synthetic_detect(dip=dip, seed=793, k=5)
    print(f"  planted hangover dip=-{dip*100:.1f}% (seed 793): mean {planted['mean']*100:+.3f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
