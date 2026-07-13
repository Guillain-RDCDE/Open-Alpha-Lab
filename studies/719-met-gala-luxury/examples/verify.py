"""Reproducible headline run for Study 719 — Met-Gala-Luxury.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached luxury / VGK tapes under
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

from met_gala_luxury import data as dt, strategy as st  # noqa: E402

print("# Met-Gala-Luxury — do LVMH/Kering/Hermes/Richemont pop around the first Monday in May?")

n_held = sum(1 for _, g, _ in dt.EVENTS if g is not None)
print(f"calendar: {len(dt.EVENTS)} listed years {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"{n_held} galas held (2000/2002 cancelled, 2020 COVID), hardcoded from Wikipedia")

if not dt.have_real():
    print("(cache miss — fetching luxury names + VGK once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Met Gala panel (4 luxury names + VGK)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]

print(f"\nevents resolved: {len(inc)} tested (of {n_held} galas held, {len(dt.EVENTS)} years "
      f"listed); {len(ev) - len(inc)} rows excluded")
print("excluded, by reason:")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")
print("  tested years:", list(inc["year"]))

print("\n# THE SIGNAL — abnormal return, day(-1) [pre-gala close] -> day(-1)+k, "
      "luxury basket minus VGK")
for label, col in (("1 week (k=5)", "ar_week"), ("1 month (k=21)", "ar_month")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  basket {label:<15s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  "
          f"t={s['t']:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event; p = share of null "
      "means at least as extreme, right-tail)")
for label, col, k, off in (("basket ar_week", "ar_week", 5, 0),
                           ("basket ar_month", "ar_month", 21, 0)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, entry_offset=off, tail="right")
    print(f"  {label:<16s}: observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) "
          f"over {pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# CONCENTRATION (third axis) — per-name one-sample t (is it hiding in one brand?)")
for h, label in (("wk", "1 week"), ("mo", "1 month")):
    pn = st.per_name_stats(ev, h)
    print(f"  {label}:")
    for _, r in pn.iterrows():
        print(f"    {r['name']:<10s} n={int(r['n'])}  mean={r['mean']*100:+.3f}%  t={r['t']:+.3f}")

print("\n# TRADABILITY — capture, day(0) [first public close] -> day(0)+k, net of costs")
for base, label in (("cap_week", "1 week"), ("cap_month", "1 month")):
    g = st.one_sample_t(inc[base + "_gross"].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  basket {label:<8s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")
pl_cap = st.placebo_pvalue(ev, prices, "cap_month_net", k=21, entry_offset=1,
                           cost_bps=5.0, tail="right")
print(f"  cap_month_net placebo: observed {pl_cap['obs']*100:+.3f}% vs placebo mean "
      f"{pl_cap['placebo_mean']*100:+.3f}%  -> p = {pl_cap['p_value']:.4f}")

print("\n# Event anatomy -- mean cumulative abnormal return by trading day (day(-1) anchor)")
cp = st.car_path(ev, prices, pre=5, post=21)
for k in (-5, -3, -1, 0, 1, 3, 5, 10, 15, 21):
    print(f"  day {k:>+3d}: {cp[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must be roughly unbiased on a null world (bump=0) and "
      "recover a planted spotlight bump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=719 + s, k=21)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for bump in (0.01, 0.02):
    p = st.synthetic_detect(bump=bump, seed=719, k=21)
    print(f"  planted bump=+{bump*100:.1f}% (seed 719): mean AR {p['mean']*100:+.3f}%  "
          f"t = {p['t']:+.2f}  (n={p['n']} synthetic events)")

print("\n# VERDICT")
print("  Signal:      NONE   -- no cut clears t>=2 (basket 1-week t=+0.74, 1-month t=+0.52,")
print("                         placebo p=0.50), and no single name clears it either.")
print("  Tradability: MIRAGE -- net-of-cost capture never clears t>=2; best is 1-week gross t=+1.07.")
print("  Just one name? NOT SUPPORTED -- strongest single house (Richemont/1-month) t=+1.00.")
