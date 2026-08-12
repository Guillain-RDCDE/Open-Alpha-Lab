"""Reproducible headline run for Study 843 — Waffle House Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / insurer /
rebuilder tapes under ``_cache/`` (fetching once on a cache miss), and always runs
the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from waffle_index import data, strategy as st  # noqa: E402

PRE, POST = 10, 20
LO, HI = 0, 20            # headline CAR horizon: landfall session .. +20 sessions

print("# Waffle House Index — do major US natural disasters move insurers & rebuilders?")
print("# (FEMA reads storm severity off whether the always-open chain closes; we ask")
print("#  whether the market reads it off ALL/TRV/PGR (payout dip) and HD/LOW (rebuild rally).)")

events = data.disaster_table()
print(f"\ndisaster table: {len(events)} major US hurricane landfalls "
      f"{events['date'].min().date()} -> {events['date'].max().date()} (hardcoded; NHC/NOAA public record)")

if not data.have_real():
    print("(cache miss — fetching SPY / ALL / TRV / PGR / HD / LOW once)")
    data.fetch()

closes = data.load_real()
spy = closes["SPY"]
for t, s in closes.items():
    print(f"  {t}: {len(s)} rows {s.index.min().date()} -> {s.index.max().date()}  fp {data.fingerprint(s)}")

ar_ins = st.basket_ar(closes, data.INSURERS, spy)
ar_reb = st.basket_ar(closes, data.REBUILDERS, spy)

print(f"\n# THE HEADLINE — per-event CAR over [+0..+{HI}] sessions, market-adjusted (vs SPY)")
ins = st.car_stats(ar_ins, events["date"], PRE, POST, LO, HI)
reb = st.car_stats(ar_reb, events["date"], PRE, POST, LO, HI)
wlo_i, whi_i = st.wilson_interval(ins["hits"], ins["n"])
print(f"  INSURERS (ALL/TRV/PGR): mean CAR {ins['mean']*100:+.2f}%  one-sample t = {ins['t']:+.2f}  "
      f"NW t = {ins['t_nw']:+.2f}  (n={ins['n']})")
print(f"     down on {ins['hits']}/{ins['n']} = {ins['hits']/ins['n']*100:.1f}% "
      f"(Wilson [{wlo_i*100:.1f}%, {whi_i*100:.1f}%])  [claim: insurers DIP]")
up_reb = int((reb["car"] > 0).sum())
wlo_r, whi_r = st.wilson_interval(up_reb, reb["n"])
print(f"  REBUILDERS (HD/LOW):    mean CAR {reb['mean']*100:+.2f}%  one-sample t = {reb['t']:+.2f}  "
      f"NW t = {reb['t_nw']:+.2f}  (n={reb['n']})")
print(f"     up on {up_reb}/{reb['n']} = {up_reb/reb['n']*100:.1f}% "
      f"(Wilson [{wlo_r*100:.1f}%, {whi_r*100:.1f}%])  [claim: rebuilders RALLY]")

print(f"\n# THE DIRECTIONAL TEST — paired (rebuilders - insurers) per-event CAR [+0..+{HI}]")
ls = st.long_short_stats(ar_ins, ar_reb, events["date"], PRE, POST, LO, HI)
b_lo, b_hi = st.block_bootstrap_ci(ls["diff"])
print(f"  rebuilders {ls['reb_mean']*100:+.2f}%  vs  insurers {ls['ins_mean']*100:+.2f}%  "
      f"-> spread {ls['mean_diff']*100:+.2f}%")
print(f"  one-sample t = {ls['t']:+.2f}   NW t = {ls['t_nw']:+.2f}   (n={ls['n']})  "
      f"boot 95% CI [{b_lo*100:+.2f}%, {b_hi*100:+.2f}%]  [claim: spread > 0]")

print(f"\n# Random-calendar placebo (20 seeds x 1,000 draws of {ls['n']} random non-disaster days)")
for name, ar, obs in (("insurers", ar_ins, ins["mean"]),
                      ("rebuilders", ar_reb, reb["mean"])):
    draws = np.concatenate([
        st.placebo_distribution(ar, ins["n"], PRE, POST, LO, HI, n_draws=1000, seed=843 + s)
        for s in range(20)])
    p = st.placebo_pvalue(obs, draws, tail="two")
    print(f"  {name:>11}: observed {obs*100:+.2f}% vs placebo {draws.mean()*100:+.2f}% "
          f"(sd {draws.std()*100:.2f}%) over {len(draws):,} draws -> two-sided p = {p:.3f}")

# long-short placebo: difference of two independent random draws
draws_ls = np.concatenate([
    st.placebo_distribution(ar_reb, ls["n"], PRE, POST, LO, HI, n_draws=1000, seed=1843 + s)
    for s in range(20)]) - np.concatenate([
    st.placebo_distribution(ar_ins, ls["n"], PRE, POST, LO, HI, n_draws=1000, seed=843 + s)
    for s in range(20)])
p_ls = st.placebo_pvalue(ls["mean_diff"], draws_ls, tail="two")
print(f"  {'spread':>11}: observed {ls['mean_diff']*100:+.2f}% vs placebo {draws_ls.mean()*100:+.2f}% "
      f"(sd {draws_ls.std()*100:.2f}%) -> two-sided p = {p_ls:.3f}")

print(f"\n# CAR PATH — insurer & rebuilder mean CAR by offset [-{PRE}..+{POST}]")
cp_i = st.car_path_stats(ar_ins, events["date"], PRE, POST)
cp_r = st.car_path_stats(ar_reb, events["date"], PRE, POST)
for k in (-5, 0, 5, 10, 20):
    print(f"  day {k:+d}: insurers CAR {cp_i.loc[k, 'car']*100:+.2f}% (t={cp_i.loc[k, 't']:+.2f})   "
          f"rebuilders CAR {cp_r.loc[k, 'car']*100:+.2f}% (t={cp_r.loc[k, 't']:+.2f})")

print("\n# THE TIMER — long rebuilders / short insurers, entered at landfall close")
print("  (dollar-neutral; one-way cost x NAV on all 4 legs + borrow on the short leg)")
for hold in (5, 10, 20):
    lg = st.timer(closes, events["date"], data.INSURERS, data.REBUILDERS, hold=hold, cost_bps=0.0)
    g = st.summarize_trades(lg, "ret_gross")
    ln = st.timer(closes, events["date"], data.INSURERS, data.REBUILDERS, hold=hold, cost_bps=5.0)
    n_ = st.summarize_trades(ln, "ret_net")
    print(f"  hold {hold:>2d}d: gross {g['mean_bps']:+8.1f} bps  net(5bps) {n_['mean_bps']:+8.1f} bps  "
          f"t(net) = {n_['t']:+.2f}  win {n_['win_rate']*100:.0f}%  (n={n_['n']})")

print("\n# Synthetic positive control — deterministic, no network")
print("  the directional detector must NOT fire on a null world (edge=0) and must recover a")
print("  planted insurer-down / rebuilder-up drift. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    cl, ev = data.synthetic_world(edge=0.0, seed=843 + s_)
    null_ts.append(st.synthetic_detect(cl, ev, data.INSURERS, data.REBUILDERS)["ls_t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
cl, ev = data.synthetic_world(edge=0.0015, seed=843)
sy = st.synthetic_detect(cl, ev, data.INSURERS, data.REBUILDERS)
print(f"  planted edge=+0.15%/day (seed 843): spread {sy['ls_mean']*100:+.2f}%  t = {sy['ls_t']:+.2f} "
      f"(insurers t={sy['ins_t']:+.2f}, rebuilders t={sy['reb_t']:+.2f})")
