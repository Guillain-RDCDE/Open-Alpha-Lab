"""Reproducible headline run for Study 733 — Kentucky-Derby-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / CHDN tapes under
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

from kentucky_derby_effect import data as dt, strategy as st  # noqa: E402

print("# Kentucky-Derby-Effect — does the Run for the Roses move the market, or CHDN?")

n_may = sum(1 for _, _, m in dt.EVENTS if m)
print(f"calendar: {len(dt.EVENTS)} Derbys {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"{n_may} on the first Saturday in May (2020 postponed to September), hardcoded from Wikipedia")

if not dt.have_real():
    print("(cache miss — fetching SPY + CHDN once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Derby panel (SPY + CHDN)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=10.0)
ev5 = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
chdn, mkt = inc[inc["leg"] == "chdn"], inc[inc["leg"] == "market"]
mkt5 = ev5[(ev5["included"]) & (ev5["leg"] == "market")]

print(f"\nevents resolved: CHDN leg {len(chdn)}/26 (full coverage — no survivorship funnel), "
      f"market seasonal leg {len(mkt)}/26 (2020 September running dropped)")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  excluded {n}x: {reason}")

print("\n# THE GAMBLING NAME — CHDN abnormal return vs SPY (beta=1 market model)")
for label, col in (("run-up (-6->-1)", "ar_runup"), ("1 week (-1->+5)", "ar_week"),
                   ("1 month (-1->+21)", "ar_month")):
    s = st.one_sample_t(chdn[col].values)
    hr = st.hit_rate(chdn[col].values)
    print(f"  {label:<18s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% (Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE MARKET SEASONAL — SPY window return, drift removed (constant-mean model)")
for label, col in (("run-up (-6->-1)", "ar_runup"), ("1 week (-1->+5)", "ar_week"),
                   ("1 month (-1->+21)", "ar_month")):
    s = st.one_sample_t(mkt[col].values)
    hr = st.hit_rate(mkt[col].values)
    print(f"  {label:<18s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% (Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event; drift-neutral null)")
pl_ch_ru = st.placebo_pvalue(ev, prices, "chdn", "ar_runup", k=5, entry_offset=0, tail="right")
print(f"  CHDN run-up (right): observed {pl_ch_ru['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_ch_ru['placebo_mean']*100:+.3f}% (sd {pl_ch_ru['placebo_sd']*100:.3f}%) -> p = {pl_ch_ru['p_value']:.4f}")
pl_ch_mo = st.placebo_pvalue(ev, prices, "chdn", "ar_month", k=21, entry_offset=0, tail="right")
print(f"  CHDN 1-month (right): observed {pl_ch_mo['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_ch_mo['placebo_mean']*100:+.3f}%  -> p = {pl_ch_mo['p_value']:.4f}")
pl_mk_wk = st.placebo_pvalue(ev, prices, "market", "ar_week", k=5, entry_offset=0, tail="left")
print(f"  market 1-week (left): observed {pl_mk_wk['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_mk_wk['placebo_mean']*100:+.3f}% (sd {pl_mk_wk['placebo_sd']*100:.3f}%) -> p = {pl_mk_wk['p_value']:.4f}")
pl_mk_mo = st.placebo_pvalue(ev, prices, "market", "ar_month", k=21, entry_offset=0, tail="left")
print(f"  market 1-month (left): observed {pl_mk_mo['obs']*100:+.3f}%  -> p = {pl_mk_mo['p_value']:.4f}")

print("\n# TRADABILITY — the costed 'trade it' timer (enter day(0), net of costs)")
for name, sub, sub5 in (("CHDN", chdn, ev5[(ev5['included']) & (ev5['leg'] == 'chdn')]),
                        ("market", mkt, mkt5)):
    for base, label in (("cap_runup", "run-up"), ("cap_week", "1 week"), ("cap_month", "1 month")):
        g = st.one_sample_t(sub[base + "_g"].values)
        n10 = st.one_sample_t(sub[base + "_n"].values)
        n5 = st.one_sample_t(sub5[base + "_n"].values)
        print(f"  {name:<6s} {label:<7s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
              f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})  "
              f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})")

pl_mk_wk_cap = st.placebo_pvalue(ev, prices, "market", "cap_week_n", k=5, entry_offset=1,
                                 cost_bps=10.0, tail="left")
print(f"\n  market 1-week net@10bps placebo (left): observed {pl_mk_wk_cap['obs']*100:+.3f}% "
      f"vs placebo mean {pl_mk_wk_cap['placebo_mean']*100:+.3f}%  -> p = {pl_mk_wk_cap['p_value']:.4f}")
print("  (the naive net t=-2.11 'clears the bar' only because costs push an already-negative")
print("   seasonal return more negative — the placebo says the dip is inside the noise)")

print("\n# Jackknife (leave-one-out) — market 1-week, the only |t| approaching 2")
x = mkt["ar_week"].values
jk = [st.one_sample_t(np.delete(x, i))["t"] for i in range(len(x))]
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk):+.3f}, {max(jk):+.3f}] across {len(x)} draws  "
      f"({sum(1 for t in jk if t <= -2)}/{len(x)} reach <=-2)")

print("\n# THIRD AXIS — 'Sell in May?' & CHDN vs the market seasonal (May years, Welch t)")
chdn_may = chdn[chdn["ran_in_may"]]
for col, label in (("ar_week", "1 week"), ("ar_month", "1 month")):
    tw = st.welch_t(chdn_may[col].values, mkt[col].values)
    print(f"  {label:<8s} CHDN {chdn_may[col].mean()*100:+.3f}% vs market {mkt[col].mean()*100:+.3f}%  "
          f"Welch t (CHDN-market) = {tw:+.3f}")

print("\n# Event anatomy — mean cumulative abnormal return by trading day (run-up start = 0)")
cp_c = st.car_path(ev, prices, "chdn")
cp_m = st.car_path(ev, prices, "market")
for k in (-5, -1, 0, 1, 5, 10, 15, 21):
    print(f"  offset {k:>3d}: CHDN {cp_c[k]*100:+.3f}%   market {cp_m[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=733 + s, k=5)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
p2 = st.synthetic_detect(bump=0.02, seed=733, k=5)
p3 = st.synthetic_detect(bump=0.03, seed=733, k=5)
print(f"  planted bump +2.0% t = {p2['t']:+.2f}   planted bump +3.0% t = {p3['t']:+.2f}  "
      f"(n={p2['n']} synthetic events)")

print("\n# VERDICT")
print("  Signal:      NONE   -- CHDN (the directly-exposed name) is flat at every horizon")
print("                        (|t|<=0.55, placebo p>=0.41) with FULL tape coverage; the market's")
print("                        only whisper (1-week -0.62%, t=-1.43) fails its drift-neutral placebo (p=0.11).")
print("  Tradability: MIRAGE -- the only |t|>=2 (market/1-week net, -2.11) is costs inflating an")
print("                        already-negative number; dies under its own placebo (p=0.11).")
print("  Sell in May? NOT SUPPORTED -- the Derby-week dip is real in sign but inside the luck cloud")
print("                        and fragile to one year (jackknife [-2.42, -1.00]).")
