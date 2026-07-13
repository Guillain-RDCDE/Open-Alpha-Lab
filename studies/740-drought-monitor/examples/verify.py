"""Reproducible headline run for Study 740 — Drought-Monitor.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY / ag-equity / grain
tapes under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
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

from drought_monitor import data, strategy as st  # noqa: E402

PRE, POST = 1, 5

print("# Drought-Monitor — does a worsening US Drought Monitor print move the ag complex?")
print("# (US Drought Monitor releases every Thursday ~8:30 ET; enter at the release-day")
print("#  close = zero look-ahead. Ag-equity basket DE/MOS/ADM/MOO, grain basket DBA/CORN/WEAT,")
print("#  abnormal = basket minus SPY.)")

events = data.drought_events()
print(f"\ndrought-escalation calendar: {len(events)} major US drought-intensification "
      f"episodes {events['date'].min().date()} -> {events['date'].max().date()} "
      f"(hardcoded Thursday USDM releases, hand-curated)")

if not data.have_real():
    print("(cache miss — fetching SPY + DE/MOS/ADM/MOO + DBA/CORN/WEAT once)")
    data.fetch()

spy, names = data.load_real()
panel = pd.DataFrame({t: s for t, s in {"SPY": spy, **names}.items()})
print(data_stamp("Drought-Monitor panel (SPY + 7 ag tickers)", panel, asof=data.AS_OF))

eq_ret, eq_cov = st.basket_returns(names, data.AG_EQUITY_TICKERS)
gr_ret, gr_cov = st.basket_returns(names, data.GRAIN_TICKERS)
spy_ret = st.daily_returns(spy)
ar_eq = st.abnormal_vs_bench(eq_ret, spy_ret)
ar_gr = st.abnormal_vs_bench(gr_ret, spy_ret)

print("\n# THE HEADLINE — print-day (day 0) abnormal return vs SPY, event window "
      f"[-{PRE}..+{POST}]")
for name, ar in (("ag-equity (DE/MOS/ADM/MOO)", ar_eq), ("grain (DBA/CORN/WEAT)", ar_gr)):
    d0 = st.day0_stats(ar, events["date"], pre=PRE, post=POST)
    print(f"  {name:<28s} n={d0['n']:2d}  mean {d0['mean']*1e4:+7.2f} bps  "
          f"t={d0['t']:+.3f}  up {d0['up']}/{d0['n']}={d0['up']/d0['n']*100:.1f}% "
          f"(Wilson [{d0['lo']*100:.1f}%, {d0['hi']*100:.1f}%])")

print(f"\n# Random-calendar placebo (20 seeds x 1,000 draws of N random non-drought days)")
for name, ar in (("ag-equity", ar_eq), ("grain", ar_gr)):
    d0 = st.day0_stats(ar, events["date"], pre=PRE, post=POST)
    draws = np.concatenate([
        st.placebo_distribution(ar, d0["n"], pre=PRE, post=POST, n_draws=1000,
                                 seed=740 + s, stat="day0") for s in range(20)])
    p = st.placebo_pvalue(d0["mean"], draws, tail="right")
    print(f"  {name:<10s}: observed {d0['mean']*1e4:+.2f} bps vs placebo mean "
          f"{draws.mean()*1e4:+.2f} bps (sd {draws.std()*1e4:.2f}) over {len(draws):,} "
          f"draws -> right-tail p = {p:.3f}")

print(f"\n# Event window — mean ag-equity abnormal return by offset [-{PRE}..+{POST}] "
      "(own one-sample t)")
cp = st.car_path_stats(ar_eq, events["date"], pre=PRE, post=POST)
for k, row in cp.iterrows():
    tag = "  <-- print day" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_ar']*1e4:+.2f} bps   CAR {row['car']*1e4:+.2f} bps  "
          f"(t={row['t']:+.2f}){tag}")

print(f"\n# Post-print drift — cumulative ag-equity abnormal return [+1..+{POST}]")
pe = st.post_event_car(ar_eq, events["date"], pre=PRE, post=POST)
print(f"  mean post-print CAR: {pe['mean']*1e4:+.2f} bps   one-sample t = {pe['t']:+.3f} "
      f"(n={pe['n']})")

print("\n# THIRD AXIS — does grain react harder than the ag-equities on the same print?")
extra = st.basket_extra_move(ar_eq, ar_gr, events["date"], pre=PRE, post=POST)
print(f"  grain day-0 AR {extra['grain_mean']*1e4:+.2f} bps  vs  ag-equity day-0 AR "
      f"{extra['equity_mean']*1e4:+.2f} bps")
print(f"  paired (grain - ag-equity) difference: {extra['mean_diff']*1e4:+.2f} bps   "
      f"one-sample t = {extra['t']:+.3f}  (n={extra['n']}, grain-ETF era events only)")

print("\n# THE TIMER — \"buy the drought\": long the basket (excess of SPY) at the print")
print("  close, hold N sessions; one round trip of one-way costs charged twice. Gross AND")
print("  net; vs the unconditional same-horizon ag-equity-minus-SPY baseline.")
for hold in (1, 5, 10, 21):
    lg = st.trade_it(eq_ret, spy_ret, events["date"], hold=hold, cost_bps=0.0)
    g = st.summarize_trade(lg, col="ret_gross")
    ln = st.trade_it(eq_ret, spy_ret, events["date"], hold=hold, cost_bps=5.0)
    n_ = st.summarize_trade(ln, col="ret_net")
    fwd = st.abnormal_vs_bench(eq_ret, spy_ret)
    base = float(fwd.rolling(hold).sum().shift(-hold).mean() * 1e4)
    print(f"  hold {hold:>2d}d: gross {g['mean_bps']:+8.2f} bps  net(5bps) {n_['mean_bps']:+8.2f} bps  "
          f"t(net)={n_['t']:+.2f}  win {n_['win_rate']*100:.0f}%  uncond. baseline {base:+.2f} bps")

print("\n# THE REGIME TEST — labelled monthly D2+ drought proxy (approximate, USDM),")
print("  high-drought vs low-drought months; regime known at month start (one shift, no")
print("  look-ahead); forward ag-equity-minus-SPY monthly return.")
proxy = data.drought_proxy()
m_ar = st.monthly_abnormal(eq_ret, spy_ret)
reg = st.regime_stats(proxy, m_ar, hi_pct=66.0, cost_bps=5.0)
print(f"  months n={reg['n']}  (high-drought threshold = top third, D2+ >= {reg['thr']:.0f}%)")
print(f"  high-drought months (n={reg['n_hi']}): mean ag abnormal {reg['hi_mean']*1e4:+.2f} bps/mo")
print(f"  other months        (n={reg['n_lo']}): mean ag abnormal {reg['lo_mean']*1e4:+.2f} bps/mo")
print(f"  Welch t (high - low) = {reg['welch_t']:+.3f}   |   high-month one-sample t = {reg['hi_t']:+.3f}")
print(f"  costed timer (long ag only in high-drought months, net 5bps): "
      f"{reg['hi_net_mean']*1e4:+.2f} bps/mo over {reg['n_entries']} entries")

print("\n# Synthetic positive control — deterministic, no network")
print("  the day0 detector must NOT fire on a null world (bump=0) and must recover a")
print("  planted print-day bump. Null checked over 20 seeds.")
null_ts = []
for s_ in range(20):
    close, ev = data.synthetic_world(bump=0.0, seed=740 + s_)
    null_ts.append(st.synthetic_detect(close, ev)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
close, ev = data.synthetic_world(bump=+0.02, seed=740)
sy = st.synthetic_detect(close, ev)
print(f"  planted bump=+2.0% (seed 740): mean {sy['mean']*1e4:+.1f} bps   t = {sy['t']:+.2f}")
