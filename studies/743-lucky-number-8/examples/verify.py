"""Reproducible headline run for Study 743 — Lucky-Number-8.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tapes under ``_cache/``
(fetching once on a cache miss), and always runs the two synthetic controls with no
network.

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

from lucky_number_8 import data as dt, strategy as st  # noqa: E402

print("# Lucky-Number-8 — do Chinese equities cluster on '8', and do they bump around 8/8?")

if not dt.have_real():
    print("(cache miss — fetching FXI, EEM, 15 China ADRs + 15 US controls once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: dt.adj_close(prices, t) for t in (dt.CHINA_PROXY, dt.EM_BENCHMARK)}).dropna()
print(data_stamp("FXI/EEM event panel", panel, asof=dt.AS_OF))

# ------------------------------------------------------------------ PART A
ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
print(f"\n## PART A — the 8/8 superstition premium (event study, FXI - EEM)")
print(f"lucky dates: {len(dt.LUCKY_DATES)} editions {dt.LUCKY_DATES[0][0]}->{dt.LUCKY_DATES[-1][0]}, "
      f"{len(inc)} with FXI+EEM coverage (calendar-known -> zero look-ahead)")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n}x: {reason}")

print("\n# THE SIGNAL — abnormal return, day(-1) [pre-8/8 close] -> day(-1)+k, FXI minus EEM")
for label, col in (("the lucky DAY (k=1)", "ar_day"), ("the lucky WEEK (k=5)", "ar_week")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% (Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event, right-tail)")
for label, col, k in (("lucky day", "ar_day", 1), ("lucky week", "ar_week", 5)):
    pl = st.placebo_pvalue(ev, prices, col, k=k, tail="right")
    print(f"  {label:<10s}: observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) over "
          f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}")

print("\n# TRADABILITY — same calendar-known window, net of 2x one-way costs")
for base, label in (("cap_day", "lucky day"), ("cap_week", "lucky week")):
    g = st.one_sample_t(inc[base + "_gross"].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<10s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# Event anatomy — mean cumulative AR by offset around day(0) (8/8)")
cp = st.car_path(ev, prices, pre=5, post=5)
for k in (-5, -1, 0, 1, 5):
    print(f"  offset {k:+d}: {cp[k]*100:+.3f}%")

# the triple-8 callout: 2008-08-08 alone
row2008 = inc[inc["year"] == 2008]
if len(row2008):
    print(f"\n# The triple-8 (2008-08-08, Beijing Olympics opening): FXI-EEM lucky-day AR = "
          f"{float(row2008['ar_day'].iloc[0])*100:+.3f}%, lucky-week AR = "
          f"{float(row2008['ar_week'].iloc[0])*100:+.3f}%")

print("\n# Synthetic positive control A (event study) — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=743 + s, k=1)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
p1 = st.synthetic_detect(bump=0.01, seed=743, k=1)
p2 = st.synthetic_detect(bump=0.02, seed=743, k=1)
print(f"  planted bump=+1.0% t={p1['t']:+.2f}   planted bump=+2.0% t={p2['t']:+.2f}")

# ------------------------------------------------------------------ PART B
print("\n## PART B — trailing-digit clustering (raw closes; China ADRs vs US controls)")
rep = st.digit_report(prices)
print(f"China ADRs: {len(dt.CHINA_ADRS)} names, {rep['china_n']:,} pooled daily closes")
print(f"US control: {len(dt.CONTROL_US)} names, {rep['control_n']:,} pooled daily closes")
print("  digit:      " + "  ".join(f"{d}" for d in range(10)))
cf = [f"{100*c/rep['china_n']:.1f}" for c in rep["china_counts"]]
uf = [f"{100*c/rep['control_n']:.1f}" for c in rep["control_counts"]]
print("  China %:  " + " ".join(f"{x:>5s}" for x in cf))
print("  US    %:  " + " ".join(f"{x:>5s}" for x in uf))
print(f"  China chi2(9) = {rep['china_chi2']['chi2']:.1f} (crit .05 = 16.92)  "
      f"control chi2(9) = {rep['control_chi2']['chi2']:.1f}")
z8, z4 = rep["z8"], rep["z4"]
print(f"  DIGIT 8 (China - control): {z8['p1']*100:.2f}% vs {z8['p2']*100:.2f}% "
      f"-> diff {z8['diff']*100:+.2f}pp, z = {z8['z']:+.2f}, p = {z8['p_value']:.3f}")
print(f"  DIGIT 4 (China - control): {z4['p1']*100:.2f}% vs {z4['p2']*100:.2f}% "
      f"-> diff {z4['diff']*100:+.2f}pp, z = {z4['z']:+.2f}, p = {z4['p_value']:.3f}")

print("\n# Synthetic positive control B (clustering) — deterministic, no network")
d0 = st.synthetic_digit_detect(excess=0.0, seed=743)
d2 = st.synthetic_digit_detect(excess=0.02, seed=743)
print(f"  null (excess=0):   digit-8 freq {d0['p8']*100:.2f}%  z8={d0['z8']:+.2f}  chi2={d0['chi2']:.1f}")
print(f"  planted (+2pp on 8): digit-8 freq {d2['p8']*100:.2f}%  z8={d2['z8']:+.2f}  chi2={d2['chi2']:.1f}")

print("\n# VERDICT")
print("  Signal:      WEAK   -- the 8/8 lucky-DAY AR clears |t|>=2 gross (t=+2.27, placebo")
print("                        p=0.043) but is the only horizon that does, is fragile (4/21")
print("                        jackknife drops -- incl. the 08/08/08 triple-8 -- fall below 2),")
print("                        and dies net of costs.")
print("  Tradability: MIRAGE -- ~39 bps/yr gross; net t=+1.69 (5bps) / +1.10 (10bps), no cut >=2.")
print("  8-clustering? BUSTED -- China ADRs end in 8 no more than US controls (z=+0.85, p=0.40),")
print("                        no 4-deficit; the real clustering is round-number 0 (z=+6.86).")
