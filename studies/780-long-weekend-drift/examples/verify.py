"""Reproducible headline run for Study 780 — Long-Weekend-Drift.

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

from long_weekend_drift import data as dt, strategy as st  # noqa: E402

print("# Long-Weekend-Drift — does SPY drift up on the pre-holiday session?")

print(f"calendar: {len(dt.EVENTS)} NYSE full-day closures {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"hardcoded from the NYSE holiday schedule (Juneteenth from 2022)")

if not dt.have_real():
    print("(cache miss — fetching SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()}).dropna()
print(data_stamp("Long-weekend panel (SPY, adjusted/total-return)", panel, asof=dt.AS_OF))

mu = st.baseline_daily(prices)
print(f"\nbaseline: SPY mean ordinary-day return = {mu*100:+.4f}% "
      f"(the 'normal day' the pre-holiday excess is measured against)")

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} holidays have SPY coverage around "
      f"the eve session")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n:2d}x {reason}")

print("\n# THE PRE-HOLIDAY DRIFT — excess return over an ordinary SPY day (gross)")
for label, col in (("holiday-eve (pre1)", "pre1"), ("3-session run-up (pre3)", "pre3")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<26s} n={s['n']:3d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE POST-HOLIDAY SESSION (reversal check) — excess return, first session after")
s = st.one_sample_t(inc["post1"].values)
hr = st.hit_rate(inc["post1"].values)
print(f"  {'post-holiday (post1)':<26s} n={s['n']:3d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}  "
      f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
      f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event)")
for label, col, k, tail in (("eve pre1", "pre1", 1, "right"),
                            ("run-up pre3", "pre3", 3, "right")):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail=tail)
    print(f"  {label:<12s} ({tail}-tail): observed {pl['obs']*100:+.4f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.4f}% (sd {pl['placebo_sd']*100:.4f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — holiday-eve (pre1) t-stat")
x = inc["pre1"].values
jk_ts = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# Sub-sample split — did the anomaly decay? (pre1)")
half = len(inc) // 2
for label, sub in (("first half (older)", inc.iloc[:half]), ("second half (recent)", inc.iloc[half:])):
    s = st.one_sample_t(sub["pre1"].values)
    print(f"  {label:<22s} n={s['n']:3d}  mean={s['mean']*100:+.4f}%  t={s['t']:+.3f}  "
          f"({sub['holiday'].iloc[0][:4]}->{sub['holiday'].iloc[-1][:4]})")

print("\n# TRADABILITY — the eve drift net of costs (calendar-known entry, zero look-ahead)")
for base, label in (("pre1", "holiday-eve"), ("pre3", "3-session run-up")):
    g = st.one_sample_t(inc[base].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<18s} gross {g['mean']*100:+.4f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.4f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.4f}% (t={n10['t']:+.2f})")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=792 + s, k=1)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.002, 0.004):
    planted = st.synthetic_detect(bump=bump, seed=792, k=1)
    print(f"  planted eve bump=+{bump*100:.1f}% (seed 792): mean excess {planted['mean']*100:+.4f}%  "
          f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  (see docs/results.md for the stamped, fingerprinted table)")
