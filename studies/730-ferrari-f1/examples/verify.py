"""Reproducible headline run for Study 730 — Ferrari-F1.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached RACE / SPY tapes under
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

from ferrari_f1 import data as dt, strategy as st  # noqa: E402

print("# Ferrari-F1 — does RACE pop the Monday after a Ferrari Grand Prix win?")

print(f"calendar: {len(dt.EVENTS)} Ferrari F1 victories 2017->2024 (winless "
      f"{', '.join(map(str, dt.WINLESS_SEASONS))}), hardcoded from STATS F1 / F1.com")

if not dt.have_real():
    print("(cache miss — fetching RACE + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Ferrari panel (RACE + SPY)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
contender = inc[inc["era"] == "contender"]
sporadic = inc[inc["era"] == "sporadic"]
inc_wk_indep = inc[~inc["weekly_overlap"]]

print(f"\nevents resolved: {len(inc)} of {len(dt.EVENTS)} wins (all postdate the "
      f"2015-10-21 RACE listing); {len(contender)} contender-era, {len(sporadic)} sporadic-era")

print("\n# THE SIGNAL — abnormal return, day(-1) [pre-race close] -> day(-1)+k, RACE minus SPY")
for label, col in (("day(0) pop (k=1)", "ar_day"), ("1 week (k=5)", "ar_week")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {label:<18s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")
s_wk_i = st.one_sample_t(inc_wk_indep["ar_week"].values)
print(f"  1 week, independent-only (drop 3 back-to-back 2nd races) n={s_wk_i['n']}  "
      f"mean={s_wk_i['mean']*100:+.3f}%  t={s_wk_i['t']:+.3f}")

print("\n# Random-calendar placebo (20 seeds x 500 draws; p = share of null means >= observed)")
pl_day = st.placebo_pvalue(ev, prices, "ar_day", k=1, entry_offset=0, tail="right")
print(f"  day(0) pop : observed {pl_day['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_day['placebo_mean']*100:+.3f}% (sd {pl_day['placebo_sd']*100:.3f}%) over "
      f"{pl_day['n_draws']:,} draws -> p = {pl_day['p_value']:.4f}")
pl_wk = st.placebo_pvalue(ev, prices, "ar_week", k=5, entry_offset=0, tail="right")
print(f"  1 week     : observed {pl_wk['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_wk['placebo_mean']*100:+.3f}% -> p = {pl_wk['p_value']:.4f}")

print("\n# TRADABILITY — capture, day(0) [first public close] -> day(0)+k, net of costs")
for base, label in (("cap_week", "1 week"),):
    g = st.one_sample_t(inc[base + "_gross"].values)
    n5 = st.one_sample_t(inc[base + "_net"].values)
    n10 = st.one_sample_t(inc10[base + "_net"].values)
    print(f"  {label:<8s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
          f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
          f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")
pl_cap = st.placebo_pvalue(ev, prices, "cap_week_net", k=5, entry_offset=1,
                           cost_bps=5.0, tail="right")
print(f"  1-week net capture placebo: observed {pl_cap['obs']*100:+.3f}% vs placebo mean "
      f"{pl_cap['placebo_mean']*100:+.3f}% -> p = {pl_cap['p_value']:.4f}")

print("\n# CONTENDER vs SPORADIC — is the whisper fan sentiment or championship fundamentals?")
for name, sub in (("contender", contender), ("sporadic", sporadic)):
    for col, label in (("ar_day", "day(0)"), ("ar_week", "1 week")):
        s = st.one_sample_t(sub[col].values)
        print(f"  {name:<10s} {label:<7s} n={s['n']:2d} mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}")
for col, label in (("ar_day", "day(0) pop"), ("ar_week", "1 week")):
    t_cs = st.welch_t(contender[col].values, sporadic[col].values)
    print(f"  Welch t (contender - sporadic), {label:<11s} = {t_cs:+.3f}")
# placebo on the one cut that clears the raw t bar: contender-era 1 week
ncex = ev[~ev["included"]]
pl_cw = st.placebo_pvalue(pd.concat([contender, ncex]), prices, "ar_week", k=5,
                          entry_offset=0, tail="right")
print(f"  contender-era 1-week placebo: observed {pl_cw['obs']*100:+.3f}% vs placebo mean "
      f"{pl_cw['placebo_mean']*100:+.3f}% -> p = {pl_cw['p_value']:.4f}  (the ONLY cut past both bars)")

print("\n# Event anatomy — mean cumulative abnormal return by trading day (day(-1) anchor)")
cp = st.car_path(ev, prices, max_k=5)
for k in range(0, 6):
    print(f"  day {k}: {cp[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must NOT fire on a null world (bump=0) and must "
      "recover a planted fan-halo bump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=730 + s, k=1)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for b in (0.01, 0.02):
    pl = st.synthetic_detect(bump=b, seed=730, k=1)
    print(f"  planted bump=+{b*100:.1f}% (seed 730): mean AR {pl['mean']*100:+.3f}%  "
          f"t = {pl['t']:+.2f}  (n={pl['n']} synthetic wins)")

print("\n# VERDICT")
print("  Signal:      NONE  -- the fan-sentiment win pop (day(0)) is a coin flip: t=+1.18,")
print("                        hit 54%, random-Monday placebo p=0.21. No immediate-reaction cut")
print("                        clears t>=2. (The one raw-significant cut is a 1-week DRIFT, not a")
print("                        pop, confined to the title-contender era -- see the third axis.)")
print("  Tradability: MIRAGE -- no edge to charge costs against; net 1-week capture +0.48% (t=0.88),")
print("                        placebo p=0.27 -- a random-window draw.")
print("  Fan-halo, or fundamentals? MISATTRIBUTED -- the only whisper (contender-era 1-week,")
print("                        t=2.73, p=0.017) lives entirely in 2017-18 when a win updated the")
print("                        title fight; the pure sporadic wins show nothing (-0.50%, t=-0.74).")
print("                        What little moves RACE tracks championship fundamentals, not fans.")
