"""Reproducible headline run for Study 731 — Wimbledon-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EWU/VGK tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

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

from wimbledon_effect import data as dt, strategy as st  # noqa: E402

print("# Wimbledon-Effect — is there a tradable UK 'summer-lull' seasonal in the FTSE?")

n_years = len(dt.contested_years())
print(f"calendar: {len(dt.WIMBLEDON)} years {dt.WIMBLEDON[0][0]}->{dt.WIMBLEDON[-1][0]}, "
      f"{n_years} contested (2020 COVID-cancelled), hardcoded from Wikipedia; "
      "each fortnight asserted Monday->Sunday, 13 days")

if not dt.have_real():
    print("(cache miss — fetching EWU + VGK once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Wimbledon panel (EWU + VGK)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc = ev[ev["included"]]
print(f"\nevents resolved: {len(inc)} of {n_years} contested fortnights have EWU + VGK "
      "coverage (calendar-known window, no look-ahead)")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")

print("\n# THE SIGNAL — fortnight window return (entry close before the first Monday -> "
      "exit close inside the fortnight)")
for label, col in (("raw EWU", "raw"), ("abnormal (EWU - VGK)", "abn")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  sd={s['sd']*100:.3f}%  "
          f"t={s['t']:+.3f}  up {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws, two-sided; p = share of |null| "
      ">= |observed|)")
for label, col in (("raw", "raw"), ("abnormal", "abn")):
    pl = st.placebo_pvalue(ev, prices, col, tail="two")
    print(f"  {label:<9s}: observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# THE VOL LULL (third axis) — realized daily-vol ratio, fortnight vs +/-25-session "
      "neighbourhood")
vl = st.vol_lull_stats(ev)
print(f"  mean log-ratio {vl['mean_log']:+.4f} (ratio {vl['mean_ratio']:.3f}), t={vl['t']:+.3f}; "
      f"quieter in {vl['quieter_k']}/{vl['n']} years = {vl['quieter_rate']*100:.1f}% "
      f"(Wilson [{vl['quieter_lo']*100:.1f}%, {vl['quieter_hi']*100:.1f}%])")

print("\n# Robustness — jackknife and the 2015 format split")
raw, abn = inc["raw"].values, inc["abn"].values
jk_r = [st.one_sample_t(np.delete(raw, i))["t"] for i in range(len(raw))]
jk_a = [st.one_sample_t(np.delete(abn, i))["t"] for i in range(len(abn))]
print(f"  raw jackknife t range [{min(jk_r):+.3f}, {max(jk_r):+.3f}]  |  "
      f"abnormal jackknife t range [{min(jk_a):+.3f}, {max(jk_a):+.3f}]")
early = inc[inc["year"] <= 2014]["raw"].values
late = inc[inc["year"] >= 2015]["raw"].values
print(f"  pre-2015 raw t={st.one_sample_t(early)['t']:+.3f} (n={len(early)})  |  "
      f"2015+ raw t={st.one_sample_t(late)['t']:+.3f} (n={len(late)})  |  "
      f"Welch (early-late) = {st.welch_t(early, late):+.3f}")

print("\n# TRADABILITY — calendar-known hold, net of costs (one-way x NAV per leg; "
      "market-neutral short pays borrow)")
for label, e, e10 in (("5 bps", ev, None), ("10 bps", ev10, None)):
    cap = st.capture_summary(e)
    for name in ("long_only", "market_neutral"):
        v = cap[name]
        print(f"  {name:<15s} @{label:<6s} gross {v['gross_mean']*100:+.3f}% (t={v['gross_t']:+.2f})  "
              f"net {v['net_mean']*100:+.3f}% (t={v['net_t']:+.2f})  win {v['win_k']}/{v['win_n']}")
# the mirror-image trade and cost drag
hold = inc["hold_days"].values
drag = 4 * 5 / 1e4 + st.BORROW_BPS_ANNUAL / 1e4 * (hold / 252.0)
rev_net = -abn - drag
r = st.one_sample_t(rev_net)
print(f"  reverse (short EWU / long VGK) net {r['mean']*100:+.3f}% (t={r['t']:+.2f}); "
      f"mean cost+borrow drag {drag.mean()*100:.3f}% per fortnight")
for col in ("long_net", "mn_net"):
    pl = st.placebo_pvalue(ev, prices, col, cost_bps=5.0, tail="two")
    print(f"  placebo {col:<8s}: observed {pl['obs']*100:+.3f}% -> p = {pl['p_value']:.4f}")

print("\n# Event anatomy — mean cumulative return by session offset from entry")
cp_r = st.car_path(ev, prices, max_k=10, col="raw")
cp_a = st.car_path(ev, prices, max_k=10, col="abn")
for k in (0, 2, 4, 6, 8, 10):
    print(f"  day {k:>2d}: raw {cp_r[k]*100:+.3f}%   abnormal {cp_a[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(0.0, 731 + s)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for b in (0.01, 0.02):
    d = st.synthetic_detect(b, 731)
    print(f"  planted seasonal = +{b*100:.0f}%: mean AR {d['mean']*100:+.3f}%  t = {d['t']:+.2f}")

print("\n# VERDICT")
print("  Signal:      NONE   -- no directional seasonal, raw or UK-specific; both cuts sit inside")
print("                         the random-window cloud, robust to jackknife and the 2015 split.")
print("  Tradability: MIRAGE -- long-only 'edge' is just market beta; the lone |t|>=2 number is a")
print("                         cost-drag LOSS on a zero-edge spread, and no trade direction pays.")
print("  A real lull? BUSTED -- fortnight realized vol equals the surrounding weeks (ratio ~1.01),")
print("                         quieter in only 10/20 years -- the 'quiet window' fails on its terms.")
