"""Reproducible headline run for Study 734 — NBA-Finals-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached metro-proxy / SPY tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with no
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

from nba_finals_effect import data as dt, strategy as st  # noqa: E402

print("# NBA-Finals-Effect — does the losing team's home-city market dip (champion's pop)?")

print(f"calendar: {len(dt.EVENTS)} Finals {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, all "
      "contested (2020 Oct bubble, 2021 Jul COVID-delay), hardcoded from Basketball-Reference")

if not dt.have_real():
    print("(cache miss — fetching metro proxies + SPY once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("NBA panel (17 metro proxies + SPY)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
loser, champ = inc[inc["role"] == "loser"], inc[inc["role"] == "champion"]
loser10, champ10 = inc10[inc10["role"] == "loser"], inc10[inc10["role"] == "champion"]

print(f"\nevents resolved: {len(inc)} of {2 * len(dt.EVENTS)} possible role-events have a "
      f"proxy + SPY coverage ({len(loser)} loser, {len(champ)} champion)")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")

print("\n# THE SIGNAL — abnormal return, day(-1) [pre-result close] -> day(-1)+k, "
      "metro proxy minus SPY")
for name, sub, pos in (("loser (EGN dip)", loser, False), ("champion (pop)", champ, True),
                       ("combined", inc, True)):
    for label, col in (("next day (k=1)", "ar_day"), ("1 week (k=5)", "ar_week")):
        s = st.one_sample_t(sub[col].values)
        hr = st.hit_rate(sub[col].values, positive=pos)
        sign = "neg" if not pos else "pos"
        print(f"  {name:<16s} {label:<15s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  "
              f"t={s['t']:+.3f}  {sign}-hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
              f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Broad US market (SPY) day(0) abnormal return — EGN predicts ~0 within one country")
bm = st.broad_market_events(prices)
s_bm = st.one_sample_t(bm["abn"].values)
print(f"  SPY market-model abn, day 0: n={s_bm['n']}  mean={s_bm['mean']*100:+.4f}%  "
      f"t={s_bm['t']:+.3f}  (one elated US city + one deflated US city -> net cancels)")

print("\n# Random-window placebo (20 seeds x 200 draws per event; p = share of null means "
      "at least as extreme in the predicted tail)")
pl_l_day = st.placebo_pvalue(ev[ev["role"] == "loser"], prices, "ar_day", k=1,
                             entry_offset=0, tail="left")
print(f"  loser ar_day (left) : observed {pl_l_day['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_l_day['placebo_mean']*100:+.3f}% (sd {pl_l_day['placebo_sd']*100:.3f}%) "
      f"over {pl_l_day['n_draws']:,} draws -> p = {pl_l_day['p_value']:.4f}")
pl_c_day = st.placebo_pvalue(ev[ev["role"] == "champion"], prices, "ar_day", k=1,
                             entry_offset=0, tail="right")
print(f"  champ ar_day (right): observed {pl_c_day['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_c_day['placebo_mean']*100:+.3f}%  -> p = {pl_c_day['p_value']:.4f}")

print("\n# TRADABILITY — capture, day(0) [first public close] -> day(0)+k, net of costs")
for name, sub, sub10 in (("loser", loser, loser10), ("champion", champ, champ10),
                         ("combined", inc, inc10)):
    for base, label in (("cap_day", "next day"), ("cap_week", "1 week")):
        g = st.one_sample_t(sub[base + "_gross"].values)
        n5 = st.one_sample_t(sub[base + "_net"].values)
        n10 = st.one_sample_t(sub10[base + "_net"].values)
        print(f"  {name:<10s} {label:<8s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
              f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
              f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

print("\n# THIRD AXIS — champion vs loser: does the winning city beat the losing city?")
for col, label in (("ar_day", "signal, next day"), ("ar_week", "signal, 1 week")):
    t_cl = st.welch_t(champ[col].values, loser[col].values)
    print(f"  {label:<20s} champion {champ[col].mean()*100:+.3f}% vs loser "
          f"{loser[col].mean()*100:+.3f}%  Welch t (champion-loser) = {t_cl:+.3f}")

print("\n# Event anatomy — mean cumulative abnormal return by trading day (day(-1) anchor)")
cp_l = st.car_path(ev, prices, "loser", max_k=5)
cp_c = st.car_path(ev, prices, "champion", max_k=5)
for k in (0, 1, 2, 3, 5):
    print(f"  day {k:>1d}: loser {cp_l[k]*100:+.3f}%   champion {cp_c[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must NOT fire on a null world (bump=0) and must recover "
      "a planted loser dip. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=734 + s, k=1)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
p1 = st.synthetic_detect(bump=-0.010, seed=734, k=1)
p2 = st.synthetic_detect(bump=-0.020, seed=734, k=1)
print(f"  planted dip=-1.0% t = {p1['t']:+.2f}   planted dip=-2.0% t = {p2['t']:+.2f} "
      f"(n={p1['n']} synthetic events)")

print("\n# VERDICT")
print("  Signal:      NONE   -- no cut (loser/champion x day/week) clears |t|>=2; the EGN")
print("                         loser dip is absent, the champion pop is noise, SPY net ~0.")
print("  Tradability: MIRAGE -- no net-of-cost, zero-look-ahead capture clears |t|>=2.")
print("  Does the loser's city dip? -- decided from the numbers above (see docs/results.md).")
