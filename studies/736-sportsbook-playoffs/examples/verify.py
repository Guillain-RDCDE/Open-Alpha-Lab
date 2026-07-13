"""Reproducible headline run for Study 736 — Sportsbook-Playoffs.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached DKNG / basket / BETZ / SPY
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

from quantlab.repro import data_stamp  # noqa: E402

from sportsbook_playoffs import data, strategy as st  # noqa: E402

RUN_UP = 10

print("# Sportsbook-Playoffs — do betting stocks rally INTO NFL playoffs / March Madness?")
print("# The claim: anticipation of a wall of betting handle lifts DKNG and the sportsbook")
print("#  basket in the ~2 weeks BEFORE the first game. Calendar-known rule -> zero look-ahead.")

events = data.event_table()
print(f"\nevent calendar: {len(events)} flagship US betting-season starts "
      f"{events['date'].min().date()} -> {events['date'].max().date()} (hardcoded; "
      f"6 NFL Wild-Card weekends + 6 March-Madness Round-of-64, 2021->2026 = the "
      f"DKNG-as-operating-company era)")

if not data.have_real():
    print("(cache miss — fetching DKNG / PENN / CZR / MGM / RSI / BETZ / SPY once)")
    data.fetch()

closes = data.load_real()
for t in ("DKNG", "PENN", "CZR", "MGM", "RSI", "BETZ", "SPY"):
    print(data_stamp(t, closes[t].to_frame("Close"), asof=data.AS_OF))

dkng = closes["DKNG"]
ret = st.daily_returns(dkng)
ar = st.abnormal_returns(ret)

print(f"\n# THE HEADLINE — DKNG run-up cumulative abnormal return over the {RUN_UP} sessions")
print(f"#  ending the session before the first game (constant-mean market model)")
r = st.run_up_stats(ar, events["date"], run_up=RUN_UP)
wl, wh = st.wilson_interval(r["hits"], r["n"])
print(f"  n = {r['n']} independent events")
print(f"  mean run-up CAR : {r['mean']*100:+.2f}%   one-sample t = {r['t']:+.3f}")
print(f"  hit rate (run-up positive) : {r['hits']}/{r['n']} = {r['hits']/r['n']*100:.1f}%  "
      f"(Wilson 95% [{wl*100:.1f}%, {wh*100:.1f}%])")
lo, hi = st.bootstrap_ci(r["per_event"])
print(f"  event-bootstrap 95% CI on the mean run-up : [{lo*100:+.2f}%, {hi*100:+.2f}%] "
      "(straddles zero)")

print(f"\n# Random-calendar placebo (20 seeds x 1,000 draws of {r['n']} random dates)")
print("#  the claim predicts a POSITIVE run-up -> falsification is the RIGHT tail")
draws = np.concatenate([
    st.placebo_distribution(ar, r["n"], run_up=RUN_UP, n_draws=1000, seed=736 + s)
    for s in range(20)
])
p_right = st.placebo_pvalue(r["mean"], draws, tail="right")
print(f"  observed {r['mean']*100:+.2f}% vs placebo mean {draws.mean()*100:+.3f}% "
      f"(sd {draws.std()*100:.2f}%) over {len(draws):,} draws -> right-tail p = {p_right:.3f}")

print("\n# By betting season — is either family alone the source of a rally?")
for f in ("NFL", "NCAA"):
    sub = events.loc[events["family"] == f, "date"]
    rr = st.run_up_stats(ar, sub, run_up=RUN_UP)
    print(f"  {f:4s}: mean run-up {rr['mean']*100:+.2f}%  t = {rr['t']:+.2f}  "
          f"(n={rr['n']}, {rr['hits']} positive)")
nfl_pe = st.run_up_stats(ar, events.loc[events["family"] == "NFL", "date"], run_up=RUN_UP)["per_event"]
ncaa_pe = st.run_up_stats(ar, events.loc[events["family"] == "NCAA", "date"], run_up=RUN_UP)["per_event"]
print(f"  Welch t (NFL - NCAA run-up): {st.welch_t(nfl_pe, ncaa_pe):+.2f} (no split by season)")

print("\n# Robustness — beta~1 market-adjusted (DKNG minus SPY) run-up")
bench_ret = st.daily_returns(closes["SPY"])
ar_madj = st.abnormal_returns(st.market_adjusted_returns(ret, bench_ret))
rm = st.run_up_stats(ar_madj, events["date"], run_up=RUN_UP)
print(f"  market-adjusted run-up : {rm['mean']*100:+.2f}%  t = {rm['t']:+.2f} "
      "(a market-wide rally is not the story either)")

print("\n# The wider betting complex — the 5-name basket and the packaged ETF")
basket, cov = st.basket_returns(closes, data.BASKET_TICKERS)
ar_b = st.abnormal_returns(basket)
rb = st.run_up_stats(ar_b, events["date"], run_up=RUN_UP)
wlb, whb = st.wilson_interval(rb["hits"], rb["n"])
print(f"  basket (DKNG/PENN/CZR/MGM/RSI, equal-weight) : {rb['mean']*100:+.2f}%  t = {rb['t']:+.2f}  "
      f"({rb['hits']}/{rb['n']} positive, Wilson [{wlb*100:.1f}%, {whb*100:.1f}%])")
covlist = [int(cov.iloc[basket.index.searchsorted(d) - 1]) for d in events["date"]]
print(f"  basket coverage at each event (of 5): {covlist} — full 5/5 for all 12 events")
ar_e = st.abnormal_returns(st.daily_returns(closes["BETZ"]))
re = st.run_up_stats(ar_e, events["date"], run_up=RUN_UP)
print(f"  BETZ ETF (Roundhill Sports Betting & iGaming) : {re['mean']*100:+.2f}%  t = {re['t']:+.2f}  "
      f"({re['hits']}/{re['n']} positive)")

print("\n# Sell-the-news? — DKNG cumulative abnormal return [0..+5] after the first game")
pe = st.post_event_stats(ar, events["date"], pre=RUN_UP, post=5)
print(f"  mean post-event CAR: {pe['mean']*100:+.2f}%   one-sample t = {pe['t']:+.3f} (n={pe['n']})")

print("\n# Event window — mean abnormal DKNG return by offset [-10..+5] (own one-sample t)")
cp = st.car_path_stats(ar, events["date"], pre=RUN_UP, post=5)
for k, row in cp.iterrows():
    tag = "  <-- first game" if k == 0 else ""
    print(f"  day {k:+d}: mean {row['mean_ar']*1e4:+.1f} bps   CAR {row['car']*1e4:+.1f} bps  "
          f"(t={row['t']:+.2f}){tag}")

print("\n# THE TIMER — buy DKNG N sessions before the first game, sell the session before it")
print("#  calendar-known rule -> zero look-ahead. One round trip, one-way cost charged twice.")
for k in (5, 10, 20):
    g = st.summarize_timer(st.run_up_timer(dkng, events["date"], run_up=k, cost_bps=0.0), "ret_gross")
    n5 = st.summarize_timer(st.run_up_timer(dkng, events["date"], run_up=k, cost_bps=5.0), "ret_net")
    n15 = st.summarize_timer(st.run_up_timer(dkng, events["date"], run_up=k, cost_bps=15.0), "ret_net")
    print(f"  run-up {k:>2d}d: gross {g['mean_bps']:+8.1f} bps  net(5) {n5['mean_bps']:+8.1f} bps  "
          f"net(15) {n15['mean_bps']:+8.1f} bps  t(net15) = {n15['t']:+.2f}  win {g['win_rate']*100:.0f}%")

print("\n# Synthetic positive control — deterministic, no network")
print("#  the run-up detector must NOT fire on a null world (bump=0) and must recover a")
print("#  planted pre-event rally. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    close, ev = data.synthetic_world(bump=0.0, seed=736 + s_)
    null_ts.append(st.synthetic_detect(close, ev)["t"])
null_ts = np.asarray(null_ts)
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f}  (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
close, ev = data.synthetic_world(bump=0.15, seed=736)
sy = st.synthetic_detect(close, ev)
print(f"  planted +15% run-up (seed 736): mean {sy['mean']*100:+.1f}%   t = {sy['t']:+.2f}")
