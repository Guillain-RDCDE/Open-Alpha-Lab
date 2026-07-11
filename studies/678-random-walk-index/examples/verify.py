"""Reproducible headline run for Study 678 — Random-Walk-Index (Poulos RWI).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily OHLC tapes under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from random_walk_index import data, strategy as st  # noqa: E402

print("# Random-Walk-Index — does an RWI-high > 1 flag actually predict the next session's return?")
print(f"periods scanned (max of RWI-high across n): {st.DEFAULT_PERIODS}  "
      f"(Poulos' own short-lookback range)")

if not data.have_real():
    print("(cache miss — fetching daily OHLC for the basket once)")
    data.fetch()

basket = data.load_basket()
for t in data.TICKERS:
    print(data_stamp(f"{t} OHLC", basket[t], asof=data.AS_OF))

spy = basket[data.HEADLINE]
df = st.day_frame(spy)
print(f"\nSPY sessions on tape: {len(df)}  {df.index.min().date()} -> {df.index.max().date()}")

print("\n# THE HEADLINE (SPY) — RWI-high>1 flag day vs no-flag day, next-session return")
s = st.trend_day_stats(df)
print(f"  flag frequency  : {s['n_flag']}/{s['n_flag']+s['n_rest']} = "
      f"{s['n_flag']/(s['n_flag']+s['n_rest'])*100:.1f}% of sessions")
print(f"  flag-day fwd ret : {s['flag_bps']:+.2f} bps   no-flag fwd ret: {s['rest_bps']:+.2f} bps")
print(f"  gap              : {s['gap_bps']:+.2f} bps   Welch t = {s['welch_t']:+.2f}   "
      f"Newey-West t (5 lags) = {s['nw_t']:+.2f}")
print(f"  hit rate (fwd ret > 0): {s['hit_up']}/{s['n_flag']} = {s['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{s['hit_lo']*100:.1f}%, {s['hit_hi']*100:.1f}%])")

print("\n# Matched-count random-day placebo (20 seeds x 1,000 draws of "
      f"{s['n_flag']} random sessions)")
pl = st.placebo_pvalue(df)
print(f"  observed flag-day mean {pl['obs']*1e4:+.2f} bps vs placebo mean "
      f"{pl['placebo_mean']*1e4:+.2f} bps (sd {pl['placebo_sd']*1e4:.2f}) over "
      f"{pl['n_draws']:,} draws -> p = {pl['p_value']:.4f}  "
      "(right-tail: how often does a random day-count-matched subset do this well)")

print("\n# THE LONG TIMER, AS AN ACTUAL BOOK (SPY, one-day execution lag)")
print("  (flag known at close t -> entered at that close -> earns close(t)->close(t+1);")
print("   costs one-way x NAV per leg on every position change)")
for cb in (0.0, 5.0, 10.0):
    bt = st.backtest(df, cost_bps=cb)
    tag = "gross" if cb == 0.0 else f"net {cb:.0f}bps"
    print(f"  {tag:>9}: total return {bt['total_return_pct']:+8.2f}%   Sharpe {bt['sharpe']:+.3f}   "
          f"(buy&hold {bt['bh_total_return_pct']:+.2f}%, Sharpe {bt['bh_sharpe']:+.3f})")
bt5 = st.backtest(df, cost_bps=5.0)
print(f"  exposure {bt5['exposure']*100:.1f}% of sessions, {bt5['n_trades']} position changes, "
      f"~{bt5['ann_cost_pct']:.2f}%/yr in turnover costs at 5 bps one-way")

print("\n# THE FAIR CONTROL — block-shuffled random entry, same exposure & turnover texture")
print("  (flag series chopped into 21-day blocks, block order randomly permuted, 20 seeds;")
print("   destroys the flag's real calendar placement, keeps days-invested count & turnover)")
rc = st.random_control_backtest(df, cost_bps=5.0, block_size=21, n_seeds=20)
print(f"  random-entry control: mean total return {rc['mean_total_return_pct']:+.2f}% "
      f"(sd {rc['sd_total_return_pct']:.2f}), mean Sharpe {rc['mean_sharpe']:+.3f} "
      f"(sd {rc['sd_sharpe']:.3f}) across {rc['n_seeds']} seeds")
p_ret = float((rc["draws_total_return_pct"] >= bt5["total_return_pct"]).mean())
p_sh = float((rc["draws_sharpe"] >= bt5["sharpe"]).mean())
print(f"  RWI timer (net 5bps): total return {bt5['total_return_pct']:+.2f}%, Sharpe {bt5['sharpe']:+.3f}")
print(f"  share of random-entry draws that BEAT the RWI timer: "
      f"{p_ret*100:.0f}% (total return), {p_sh*100:.0f}% (Sharpe)")

print("\n# CROSS-INSTRUMENT — SPY, QQQ, IWM, DIA, GLD (2005-01 -> 2026-06)")
xi = st.cross_instrument_stats(basket)
print(f"  {'ticker':>6} {'n_flag':>7} {'flag_bps':>9} {'rest_bps':>9} {'gap_bps':>8} "
      f"{'welch_t':>8} {'nw_t':>7} {'hit%':>6}")
for t, r in xi["per_ticker"].items():
    print(f"  {t:>6} {r['n_flag']:>7} {r['flag_bps']:>9.2f} {r['rest_bps']:>9.2f} "
          f"{r['gap_bps']:>8.2f} {r['welch_t']:>8.2f} {r['nw_t']:>7.2f} {r['hit_rate']*100:>5.1f}%")
pooled = xi["pooled"]
print(f"  pooled (n={pooled['n_flag']+pooled['n_rest']}): flag {pooled['flag_bps']:+.2f} bps vs "
      f"rest {pooled['rest_bps']:+.2f} bps   Welch t = {pooled['welch_t']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  two-regime (trend/chop) Markov world, p(stay)=0.99, same sigma both regimes.")
print("  the detector must NOT fire on the null (edge=0) and must recover a planted edge.")
null_ts = []
for s_ in range(20):
    ohlc = data.synthetic_world(edge=0.0, seed=678 + s_)
    null_ts.append(st.synthetic_detect(ohlc)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")
ohlc = data.synthetic_world(edge=0.006, seed=678)
sy = st.synthetic_detect(ohlc)
print(f"  planted edge=+0.006/day in the trend regime (seed 678): flag-day mean "
      f"{sy['flag_bps']:+.2f} bps vs no-flag {sy['rest_bps']:+.2f} bps   "
      f"Welch t = {sy['welch_t']:+.2f}   hit rate {sy['hit_rate']*100:.1f}%")
