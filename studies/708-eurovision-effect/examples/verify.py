"""Reproducible headline run for Study 708 — Eurovision-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached country-ETF / VGK tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

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

from eurovision_effect import data as dt, strategy as st  # noqa: E402

print("# Eurovision-Effect — does the winner's (or host's) stock market get a feel-good bump?")

n_years = sum(1 for _, fd, *_ in dt.EVENTS if fd is not None)
print(f"calendar: {len(dt.EVENTS)} editions {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      f"{n_years} contested (2020 COVID-cancelled), hardcoded from Wikipedia")

if not dt.have_real():
    print("(cache miss — fetching country ETFs + VGK once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Eurovision panel (14 country ETFs + VGK)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
ev10 = st.build_event_table(prices, cost_bps=10.0)
inc, inc10 = ev[ev["included"]], ev10[ev10["included"]]
winner, host = inc[inc["role"] == "winner"], inc[inc["role"] == "host"]
winner10, host10 = inc10[inc10["role"] == "winner"], inc10[inc10["role"] == "host"]

print(f"\nevents resolved: {len(inc)} of {2 * len(dt.EVENTS)} possible role-events have a "
      f"tradable country ETF + VGK coverage ({len(winner)} winner, {len(host)} host)")
print("excluded, by reason:")
for reason, n in ev[~ev["included"]]["reason"].value_counts().items():
    print(f"  {n:2d}x {reason}")

print("\n# THE SIGNAL — abnormal return, day(-1) [pre-final close] -> day(-1)+k, "
      "country ETF minus VGK")
for name, sub in (("winner", winner), ("host", host), ("combined (winner+host)", inc)):
    for label, col in (("1 week (k=5)", "ar_week"), ("1 month (k=21)", "ar_month")):
        s = st.one_sample_t(sub[col].values)
        hr = st.hit_rate(sub[col].values)
        print(f"  {name:<22s} {label:<14s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  "
              f"t={s['t']:+.3f}  hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
              f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# Random-window placebo (20 seeds x 200 draws per event; p = share of null "
      "means at least as extreme)")
pl_w_mo = st.placebo_pvalue(ev[ev["role"] == "winner"], prices, "ar_month", k=21,
                            entry_offset=0, tail="right")
print(f"  winner ar_month : observed {pl_w_mo['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_w_mo['placebo_mean']*100:+.3f}% (sd {pl_w_mo['placebo_sd']*100:.3f}%) "
      f"over {pl_w_mo['n_draws']:,} draws -> p = {pl_w_mo['p_value']:.4f}")
pl_c_mo = st.placebo_pvalue(ev, prices, "ar_month", k=21, entry_offset=0, tail="right")
print(f"  combined ar_month: observed {pl_c_mo['obs']*100:+.3f}%  vs placebo mean "
      f"{pl_c_mo['placebo_mean']*100:+.3f}%  -> p = {pl_c_mo['p_value']:.4f}")

print("\n# Jackknife (leave-one-out) — winner ar_month, the only cut clearing t>=2")
x = winner["ar_month"].values
years, countries = winner["year"].values, winner["country"].values
jk_ts = []
for i in range(len(x)):
    s = st.one_sample_t(np.delete(x, i))
    jk_ts.append(s["t"])
print(f"  full-sample t = {st.one_sample_t(x)['t']:+.3f}  |  jackknife t range "
      f"[{min(jk_ts):+.3f}, {max(jk_ts):+.3f}] across {len(x)} leave-one-out draws")

print("\n# TRADABILITY — capture, day(0) [first public close] -> day(0)+k, net of costs")
for name, sub, sub10 in (("winner", winner, winner10), ("host", host, host10),
                         ("combined", inc, inc10)):
    for base, label in (("cap_week", "1 week"), ("cap_month", "1 month")):
        g = st.one_sample_t(sub[base + "_gross"].values)
        n5 = st.one_sample_t(sub[base + "_net"].values)
        n10 = st.one_sample_t(sub10[base + "_net"].values)
        print(f"  {name:<10s} {label:<8s} gross {g['mean']*100:+.3f}% (t={g['t']:+.2f})  "
              f"net@5bps {n5['mean']*100:+.3f}% (t={n5['t']:+.2f})  "
              f"net@10bps {n10['mean']*100:+.3f}% (t={n10['t']:+.2f})")

pl_h_capwk = st.placebo_pvalue(ev[ev["role"] == "host"], prices, "cap_week_net", k=5,
                               entry_offset=1, cost_bps=5.0, tail="left")
print(f"\n  host cap_week_net placebo (left-tail): observed {pl_h_capwk['obs']*100:+.3f}% "
      f"vs placebo mean {pl_h_capwk['placebo_mean']*100:+.3f}%  -> p = "
      f"{pl_h_capwk['p_value']:.4f}  (the naive t looked significant; the placebo says it isn't)")

print("\n# THIRD AXIS -- winning vs hosting: does it matter which one you did?")
for col, label in (("ar_month", "signal, 1 month"), ("cap_month_net", "capture, 1 month net")):
    t_wh = st.welch_t(winner[col].values, host[col].values)
    print(f"  {label:<22s} winner {winner[col].mean()*100:+.3f}% vs host {host[col].mean()*100:+.3f}%  "
          f"Welch t (winner-host) = {t_wh:+.3f}")

print("\n# Event anatomy -- mean cumulative abnormal return by trading day (day(-1) anchor)")
cp_w = st.car_path(ev, prices, "winner", max_k=21)
cp_h = st.car_path(ev, prices, "host", max_k=21)
for k in (0, 5, 10, 15, 21):
    print(f"  day {k:>2d}: winner {cp_w[k]*100:+.3f}%   host {cp_h[k]*100:+.3f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must NOT fire on a null world (bump=0) and must "
      "recover a planted national-pride bump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=708 + s, k=21)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
planted = st.synthetic_detect(bump=0.02, seed=708, k=21)
print(f"  planted bump=+2.0% (seed 708): mean AR {planted['mean']*100:+.3f}%  "
      f"t = {planted['t']:+.2f}  (n={planted['n']} synthetic events)")

print("\n# VERDICT")
print("  Signal:      WEAK  -- one of four cuts (winner, 1-month) nominally clears t>=2 with a")
print("                        placebo p<0.05, but does not replicate at 1-week, for hosts, or")
print("                        pooled, and only 12-13 of 25 richer-Europe events have any tradable")
print("                        vehicle at all.")
print("  Tradability: MIRAGE -- net-of-cost capture never clears t>=2 in any cut; the strongest")
print("                        (winner, 1-month, net) is t=+1.78, placebo p=0.058.")
print("  Winning beats hosting? MIXED -- winner point estimates beat host's in both horizons, but")
print("                        neither survives to a bankable, replicated edge.")
