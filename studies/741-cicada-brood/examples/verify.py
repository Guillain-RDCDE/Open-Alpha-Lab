"""Reproducible headline run for Study 741 — Cicada-Brood.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY total-return tape under
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

from quantlab.repro import data_stamp  # noqa: E402

from cicada_brood import data, strategy as st  # noqa: E402

print("# Cicada-Brood — do periodical-cicada emergence springs move the S&P 500?")
print("# (A deliberately silly, fixed-calendar spurious-pattern demo, stated as such.)")

broods = data.brood_table()
ev_years = data.brood_years()
print(f"\nbrood table: {len(broods)} mapped periodical-cicada emergences, "
      f"{broods['year'].min()}->{broods['year'].max()}; {len(ev_years)} distinct "
      f"emergence YEARS (the event unit). A brood emerges somewhere in "
      f"{len(ev_years)}/{broods['year'].max()-broods['year'].min()+1} years of the span.")

if not data.have_real():
    print("(cache miss — fetching SPY once)")
    data.fetch()

spy = data.load_real()
print(data_stamp("SPY total-return close", spy.to_frame("Close"), asof=data.AS_OF))

ret = st.daily_returns(spy)
ar = st.abnormal_returns(ret)
pool = data.all_years(spy)
K, KS = data.WINDOW_K, data.WINDOW_K_SHORT
print(f"\nbaseline / placebo pool: {len(pool)} calendar years with a full {K}-session "
      f"May-June window on the SPY tape ({pool[0]}->{pool[-1]})")

# --------------------------------------------------------------------------- #
print(f"\n# THE HEADLINE — mean cicada-spring window return (May 1 -> +{K} sessions, ~2mo),"
      " total-return SPY")
for label, years in (("all notable broods", ev_years),
                     ("17-year broods only", data.brood_years(cycle=17)),
                     ("famous marquee only", data.brood_years(famous_only=True))):
    evt = st.build_event_table(spy, ar, years, k=K)
    s_raw = st.one_sample_t(evt["raw_ret"].to_numpy())
    s_abn = st.one_sample_t(evt["abn_car"].to_numpy())
    hr = st.hit_rate(evt["raw_ret"].to_numpy())
    print(f"  {label:<22s} n={s_raw['n']:2d}  raw {s_raw['mean']*100:+.3f}%  "
          f"abn-CAR {s_abn['mean']*100:+.3f}% (t={s_abn['t']:+.2f})  "
          f"up {hr['k']}/{hr['n']}={hr['rate']*100:.0f}% (Wilson [{hr['lo']*100:.0f}%,{hr['hi']*100:.0f}%])")

evt = st.build_event_table(spy, ar, ev_years, k=K)
base_bps = st.unconditional_spring_baseline(spy, pool, k=K)
obs_raw = float(evt["raw_ret"].mean())
print(f"\n  unconditional all-year May-June baseline: {base_bps:+.1f} bps "
      f"(every spring, {len(pool)} years)")
print(f"  cicada-spring mean: {obs_raw*1e4:+.1f} bps  ->  cicada minus baseline "
      f"{obs_raw*1e4 - base_bps:+.1f} bps")

# --------------------------------------------------------------------------- #
print(f"\n# Random-year placebo (20 seeds x 1,000 draws of {len(ev_years)} random years; "
      "same May-June window controls the season)")
pl = st.random_year_placebo(spy, pool, n_events=len(ev_years), k=K,
                            n_seeds=20, n_draws_per_seed=1000, base_seed=741, tail="right")
p_right = st.placebo_pvalue(obs_raw, pl["means"], tail="right")
print(f"  observed {obs_raw*100:+.3f}% vs placebo mean {pl['placebo_mean']*100:+.3f}% "
      f"(sd {pl['placebo_sd']*100:.3f}%) over {pl['n_draws']:,} draws -> right-tail "
      f"p = {p_right:.3f}")

# --------------------------------------------------------------------------- #
print(f"\n# 1-month window cross-check (May 1 -> +{KS} sessions)")
evt_s = st.build_event_table(spy, ar, ev_years, k=KS)
s_abn_s = st.one_sample_t(evt_s["abn_car"].to_numpy())
print(f"  all notable broods, abn-CAR {s_abn_s['mean']*100:+.3f}% (t={s_abn_s['t']:+.2f}, "
      f"n={s_abn_s['n']})")

# --------------------------------------------------------------------------- #
print("\n# Welch — cicada springs vs non-cicada springs (same May-June window)")
non_years = [y for y in pool if y not in set(ev_years)]
w_all = st.all_year_windows(spy, pool, k=K)
a = np.array([w_all[y] for y in ev_years if y in w_all.index])
b = np.array([w_all[y] for y in non_years if y in w_all.index])
print(f"  cicada springs (n={len(a)}) mean {a.mean()*100:+.3f}%  vs  "
      f"non-cicada springs (n={len(b)}) mean {b.mean()*100:+.3f}%  "
      f"Welch t (cicada-non) = {st.welch_t(a, b):+.3f}")

# --------------------------------------------------------------------------- #
print("\n# Event anatomy — mean cumulative abnormal return by offset (May-1 anchor)")
cp = st.car_path(spy, ar, ev_years, k=K)
for k in (0, 10, 21, 31, K):
    print(f"  offset {k:>2d}: CAR {cp.loc[k, 'car']*100:+.3f}%  (t={cp.loc[k, 't']:+.2f})")

# --------------------------------------------------------------------------- #
print("\n# THE TIMER — long SPY over the cicada-spring window only (pre-schedulable; zero")
print("  look-ahead by construction), one round trip of one-way costs charged twice vs NAV,")
print("  compared to the unconditional all-year spring baseline over the same tape.")
for cost in (0.0, 5.0, 10.0):
    lg = st.spring_timer(spy, ev_years, k=K, cost_bps=cost)
    col = "ret_gross" if cost == 0.0 else "ret_net"
    s = st.summarize_timer(lg, col=col)
    tag = "gross" if cost == 0.0 else f"net@{cost:.0f}bps"
    print(f"  {tag:<11s} mean {s['mean_bps']:+7.1f} bps  win {s['win_rate']*100:.0f}%  "
          f"t(vs 0) = {s['t']:+.2f}   (baseline every-spring {base_bps:+.1f} bps)")
print("  NOTE: that t(vs 0) is just SPY's ordinary 2-month bull drift (beta), not an edge.")
print("  The HONEST tradability test is the EXCESS over the every-spring baseline (alpha):")
for cost in (0.0, 5.0, 10.0):
    ex = st.excess_over_baseline(spy, ev_years, pool, k=K, cost_bps=cost)
    tag = "gross" if cost == 0.0 else f"net@{cost:.0f}bps"
    print(f"    excess {tag:<9s} {ex['excess_bps']:+7.1f} bps  t = {ex['t']:+.2f}  (n={ex['n']})")

# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control — deterministic, no network")
print("  the abn-CAR detector must NOT fire on a null world (bump=0) and must recover a")
print("  planted emergence-spring bump. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    close, em = data.synthetic_world(bump=0.0, seed=741 + s_)
    null_ts.append(st.synthetic_detect(close, em, k=K)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
close, em = data.synthetic_world(bump=0.04, seed=741)
sy = st.synthetic_detect(close, em, k=K)
print(f"  planted bump=+4.0% (seed 741): abn-CAR {sy['mean']*100:+.2f}%  t = {sy['t']:+.2f}  "
      f"(n={sy['n']} synthetic emergence springs)")

print("\n# VERDICT")
print("  Signal:      NONE   -- cicada springs are statistically indistinguishable from")
print("                         random springs; placebo p ~ 0.5, |t| well under 2.")
print("  Tradability: MIRAGE -- the calendar overlay has no edge over just holding; a")
print("                         fully-foreseeable signal that still buys you nothing.")
print("  Cicada indicator? BUSTED -- a brood emerges in most years anyway; the pattern is noise.")
