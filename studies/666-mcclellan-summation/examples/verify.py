"""Reproducible headline run for Study 666 — McClellan Summation Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached breadth-basket tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from mcclellan_summation import data as d, strategy as st  # noqa: E402

AS_OF = "2026-06-30"

print("# McClellan Summation Index — does the running integral of breadth time SPY?")

if not d.have_real():
    print("(cache miss — fetching the breadth basket + SPY once)")
    d.fetch_all(end="2026-07-01")
if not d.have_real(d.BREADTH_FALLBACK):
    d.fetch_all(tickers=d.BREADTH_FALLBACK, end="2026-07-01")

close = d.load_real(d.TRADED)
close = close[close.index <= AS_OF]["close"]
net = d.load_breadth(asof=AS_OF).reindex(close.index).dropna()
close = close.reindex(net.index)
print(data_stamp("SPY close", close.to_frame("close"), asof=AS_OF))
print(data_stamp("breadth basket (net advances)", net.to_frame("net_adv"), asof=AS_OF))
print(f"breadth basket: {d.breadth_members()} ({len(d.breadth_members())} names)")

summ = st.summation_index(net)
print(f"\nSummation Index range on the real tape: [{summ.min():.2f}, {summ.max():.2f}]  "
      f"(warm-up = {st._warmup(len(close))} sessions dropped)")

print("\n# THE HEADLINE — zero-cross events (the textbook regime-turn signal)")
raw_up_all = st.zero_cross_dates(summ, "up", apply_warmup=False)
raw_dn_all = st.zero_cross_dates(summ, "down", apply_warmup=False)
up_post = st.zero_cross_dates(summ, "up")
dn_post = st.zero_cross_dates(summ, "down")
print(f"  raw zero-up-crosses (no warm-up filter): {len(raw_up_all)}  {list(raw_up_all.date)}")
print(f"  raw zero-down-crosses (no warm-up filter): {len(raw_dn_all)}")
print(f"  post-warm-up zero-up-crosses: {len(up_post)}  |  post-warm-up zero-down-crosses: {len(dn_post)}")
print("  -> the un-rebased cumulative sum never revisits zero after its first few sessions"
      " on this 2005-2026 basket: the textbook zero-cross rule is STRUCTURALLY INERT here.")

print("\n# Extreme-threshold events — causal rolling z-score, the scale-appropriate")
print("  analog of the literature's +/-500 level (a small correlated ETF basket cannot")
print("  reproduce a full-NYSE-exchange numeric level)")
for direction in ("up", "down"):
    ex = st.cross_experiment(close, summ, direction=direction, kind="extreme", level=1.0, cost_bps=1.0)
    print(f"\n  {direction}-cross (z {'>' if direction=='up' else '<'} "
          f"{'+1' if direction=='up' else '-1'}): n = {ex['n_entries']}")
    for h in st.HORIZONS:
        row = ex["by_h"][h]
        print(f"    H={h:>2}d  trigger {row['gross']['mean_bps']:+7.1f} bps (t={row['gross']['t']:+.2f})"
              f"  random {row['random']['mean_bps']:+7.1f} bps  delta {row['delta_bps']:+7.1f} bps"
              f"  Welch t = {row['welch_t']:+.2f}")

print("\n# Shuffled-breadth placebo — extreme up-cross, H=20d, 500 draws")
pl_ext = st.shuffled_breadth_placebo(close, net, horizon=20, direction="up", kind="extreme", n_draws=500)
print(f"  observed {pl_ext['obs']*1e4:+.1f} bps  ->  p = {pl_ext['p_value']:.3f}  "
      f"(n={pl_ext['n_draws']} valid draws)")

print("\n# THE LONG/FLAT REGIME TIMER — the headline tradability test")
print("  (one execution lag: yesterday's close regime sets today's exposure; cost 5 bps")
print("   one-way x NAV per regime switch)")
reg_lvl = st.regime_from_level(summ)
rb_lvl = st.regime_backtest(close, reg_lvl, cost_bps=5.0)
print(f"  textbook zero-level regime: {rb_lvl['n_switches']} switches over "
      f"{rb_lvl['n_years']:.1f} years -> frac_long = {rb_lvl['frac_long']:.2f} "
      f"(structurally == buy-and-hold; excess = {rb_lvl['excess_mean_bps_day']:+.3f} bps/day)")

reg_z = st.regime_from_zscore(summ)
rb_z = st.regime_backtest(close, reg_z, cost_bps=5.0)
print(f"  z-score regime (bull while Summ > its trailing 1y mean): "
      f"{rb_z['n_switches']} switches, {rb_z['switches_per_year']:.1f}/yr")
print(f"    timed   : CAGR {rb_z['timed']['cagr']*100:+.2f}%  vol {rb_z['timed']['ann_vol']*100:.2f}%  "
      f"Sharpe {rb_z['timed']['sharpe']:.2f}  (long {rb_z['frac_long']*100:.0f}% of days)")
print(f"    buy&hold: CAGR {rb_z['buy_hold']['cagr']*100:+.2f}%  vol {rb_z['buy_hold']['ann_vol']*100:.2f}%  "
      f"Sharpe {rb_z['buy_hold']['sharpe']:.2f}")
print(f"    excess (timed - buy&hold): {rb_z['excess_mean_bps_day']:+.3f} bps/day  "
      f"HAC t = {rb_z['excess_hac_t']:+.2f}")

print("\n# Regime-timer shuffled-breadth placebo (500 draws)")
pl_reg = st.shuffled_breadth_regime_placebo(close, net, cost_bps=5.0, n_draws=500)
print(f"  observed HAC t = {pl_reg['obs_t']:+.2f}  ->  two-sided p = {pl_reg['p_value']:.3f}  "
      f"(n={pl_reg['n_draws']} valid draws)")

print("\n# Robustness — alternate EMA spans (fast, slow)")
for fast, slow in ((10, 20), (19, 39), (25, 50)):
    s2 = st.summation_index(net, fast, slow)
    r2 = st.regime_from_zscore(s2)
    rb2 = st.regime_backtest(close, r2, cost_bps=5.0, warmup=st._warmup(len(close), slow))
    print(f"  spans {fast:>2}/{slow:>2}: {rb2['n_switches']:>3} switches, "
          f"excess {rb2['excess_mean_bps_day']:+.3f} bps/day, HAC t = {rb2['excess_hac_t']:+.2f}")

print("\n# Robustness — alternate breadth basket (5-ticker fallback SPY/QQQ/IWM/DIA/GLD)")
net5 = d.load_breadth(members=d.BREADTH_FALLBACK, asof=AS_OF).reindex(close.index).dropna()
close5 = close.reindex(net5.index)
summ5 = st.summation_index(net5)
reg5 = st.regime_from_zscore(summ5)
rb5 = st.regime_backtest(close5, reg5, cost_bps=5.0)
print(f"  5-ticker basket: {rb5['n_switches']} switches, excess {rb5['excess_mean_bps_day']:+.3f} bps/day, "
      f"HAC t = {rb5['excess_hac_t']:+.2f}  (zero-level crosses: "
      f"{len(st.zero_cross_dates(summ5, 'up'))} up / {len(st.zero_cross_dates(summ5, 'down'))} down)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the regime-timer detector must NOT fire on a null world (edge=0) and must")
print("  recover a planted regime effect. Null checked over 20 seeds (never one stream).")
null_lvl, null_z = [], []
for s_ in range(20):
    bars, _ = d.synthetic_panel(n_days=4000, edge=0.0, seed=666 + s_)
    null_lvl.append(st.synthetic_regime_detect(bars, bars["net_adv"], kind="level")["excess_hac_t"])
    null_z.append(st.synthetic_regime_detect(bars, bars["net_adv"], kind="zscore")["excess_hac_t"])
null_lvl = np.asarray(null_lvl, dtype=float)
null_z = np.asarray(null_z, dtype=float)
print(f"  null (edge=0), level regime,  20 seeds: mean t = {np.nanmean(null_lvl):+.2f} "
      f"(sd {np.nanstd(null_lvl, ddof=1):.2f}), |t|>=2 in {(np.abs(null_lvl) >= 2).sum()}/20 seeds")
print(f"  null (edge=0), zscore regime, 20 seeds: mean t = {np.nanmean(null_z):+.2f} "
      f"(sd {np.nanstd(null_z, ddof=1):.2f}), |t|>=2 in {(np.abs(null_z) >= 2).sum()}/20 seeds")

bars, _ = d.synthetic_panel(n_days=4000, edge=0.6, seed=666)
sy_lvl = st.synthetic_regime_detect(bars, bars["net_adv"], kind="level")
sy_z = st.synthetic_regime_detect(bars, bars["net_adv"], kind="zscore")
print(f"  planted edge=0.6 (seed 666): level regime t = {sy_lvl['excess_hac_t']:+.2f} "
      f"({sy_lvl['excess_mean_bps_day']:+.2f} bps/day) | zscore regime t = {sy_z['excess_hac_t']:+.2f} "
      f"({sy_z['excess_mean_bps_day']:+.2f} bps/day)")
