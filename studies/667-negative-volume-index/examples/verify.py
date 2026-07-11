"""Reproducible headline run for Study 667 — Negative Volume Index (Fosback).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^GSPC / SPY tapes under
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

from negative_volume_index import data, strategy as st  # noqa: E402

print("# Negative Volume Index — does NVI above its 1-year EMA mean ~96% bull-market odds?")

if not data.have_real():
    print("(cache miss — fetching ^GSPC / SPY once)")
    data.fetch()

gspc, spy = data.load_real()
print(data_stamp("^GSPC OHLCV", gspc, asof=data.AS_OF))
print(data_stamp("SPY OHLCV (total-return)", spy, asof=data.AS_OF))

n_gspc = st.nvi(gspc["Close"], gspc["Volume"])
e_gspc = st.nvi_ema(n_gspc, span=data.EMA_SPAN)
reg_gspc = st.regime(n_gspc, e_gspc)
print(f"\n^GSPC NVI built {gspc.index.min().date()} -> {gspc.index.max().date()} "
      f"({len(gspc):,} sessions); regime live from "
      f"{reg_gspc.dropna().index.min().date()} ({int((~reg_gspc.isna()).sum()):,} days), "
      f"{int((reg_gspc == True).sum()):,} NVI>EMA / {int((reg_gspc == False).sum()):,} NVI<EMA")  # noqa: E712

print("\n# THE HEADLINE — Fosback's own annual framing (calendar-year replication)")
at = st.annual_bull_test(gspc["Close"], reg_gspc)
print(f"  {at['n_years']} calendar years, {at['lo_year']} -> {at['hi_year']} "
      f"(NVI state read at each year-end predicts the FOLLOWING year's return)")
lo, hi = at["p_on_ci"]
print(f"  P(year up | NVI>EMA at prior year-end) = {at['p_on']*100:.1f}%  "
      f"(n={at['n_on']}, Wilson [{lo*100:.1f}%, {hi*100:.1f}%])")
lo, hi = at["p_off_ci"]
print(f"  P(year up | NVI<EMA at prior year-end) = {at['p_off']*100:.1f}%  "
      f"(n={at['n_off']}, Wilson [{lo*100:.1f}%, {hi*100:.1f}%])")
lo, hi = at["p_all_ci"]
print(f"  P(year up)  unconditional  base rate  = {at['p_all']*100:.1f}%  "
      f"(n={at['n_years']}, Wilson [{lo*100:.1f}%, {hi*100:.1f}%])")
print(f"  gap (NVI>EMA minus unconditional base rate): {at['gap_on_vs_all']*100:+.1f} pp")

pl = st.annual_placebo(at["returns"], at["regime"])
print(f"\n  Label-shuffle placebo ({pl['n_draws']:,} draws, same duty cycle): "
      f"observed gap {pl['obs_gap']*100:+.1f} pp vs placebo mean "
      f"{pl['placebo_gap_mean']*100:+.2f} pp (sd {pl['placebo_gap_sd']*100:.2f} pp) "
      f"-> p = {pl['p_value']:.3f}")

print("\n# Higher-power daily cross-check — forward returns by regime, 3 horizons")
print("  (naive Welch t ignores overlap and is a KNOWN TRAP; Newey-West lag=horizon is")
print("   the number the REAL bar is judged against)")
for h in (21, 63, 252):
    s = st.horizon_split(gspc["Close"], reg_gspc, h)
    lo_on, hi_on = s["hit_on_ci"]
    lo_off, hi_off = s["hit_off_ci"]
    print(f"  {h:>3d}d: on {s['mean_on']*100:+.3f}% (n={s['n_on']:,}) vs "
          f"off {s['mean_off']*100:+.3f}% (n={s['n_off']:,})  "
          f"naive Welch t={s['welch_t']:+.2f}  Newey-West t={s['nw_t']:+.2f}")
    print(f"        hit-rate (overlap-inflated CI, informational only): "
          f"on {s['hit_on']*100:.1f}% [{lo_on*100:.1f},{hi_on*100:.1f}]  "
          f"off {s['hit_off']*100:.1f}% [{lo_off*100:.1f},{hi_off*100:.1f}]")

print("\n# THIRD AXIS — the costed long/flat timer on SPY (tradable proxy)")
n_spy = st.nvi(spy["Close"], spy["Volume"])
e_spy = st.nvi_ema(n_spy, span=data.EMA_SPAN)
reg_spy = st.regime(n_spy, e_spy)
pos_spy = reg_spy.fillna(False).astype(float)
print(f"  SPY NVI regime: signal ON (long) {pos_spy.mean()*100:.1f}% of "
      f"{len(spy):,} sessions {spy.index.min().date()} -> {spy.index.max().date()}")

for cb in (0.0, 5.0, 10.0):
    bt = st.backtest(spy["Close"], pos_spy, cost_bps=cb)
    su = st.summarize(bt)
    print(f"  cost={cb:>4.1f}bps: Sharpe net {su['sharpe_net']:.3f} (B&H {su['bh_sharpe']:.3f})  "
          f"CAGR net {su['cagr_net']*100:+.2f}% (B&H {su['bh_cagr']*100:+.2f}%)  "
          f"spread {su['mean_spread_bps']:+.3f} bps/day  HAC t={su['spread_t']:+.2f}")

bt5 = st.backtest(spy["Close"], pos_spy, cost_bps=5.0)
su5 = st.summarize(bt5)
print(f"  switches: {su5['n_switches']} over {su5['n_days']/252:.1f} yr "
      f"({su5['switches_per_yr']:.2f}/yr)  max DD net {su5['maxdd_net']*100:.1f}% "
      f"(B&H {su5['bh_maxdd']*100:.1f}%)")

perm = st.permutation_pvalue(spy["Close"].pct_change(), pos_spy)
print(f"  circular-shift placebo (2,000 draws, gross spread): observed "
      f"{perm['observed_spread_bps']:+.3f} bps vs placebo mean "
      f"{perm['placebo_mean_bps']:+.3f} bps -> p = {perm['p_value']:.3f}")

print("\n# Robustness — sample-half split (SPY, own NVI, cost=5bps)")
half = len(spy) // 2
for lbl, seg in (("H1", spy.iloc[:half]), ("H2", spy.iloc[half:])):
    n_s = st.nvi(seg["Close"], seg["Volume"])
    e_s = st.nvi_ema(n_s, span=data.EMA_SPAN)
    reg_s = st.regime(n_s, e_s)
    pos_s = reg_s.fillna(False).astype(float)
    bt_s = st.backtest(seg["Close"], pos_s, cost_bps=5.0)
    su_s = st.summarize(bt_s)
    print(f"  {lbl} ({seg.index.min().date()} -> {seg.index.max().date()}): "
          f"Sharpe net {su_s['sharpe_net']:.3f} (B&H {su_s['bh_sharpe']:.3f})  "
          f"spread {su_s['mean_spread_bps']:+.3f} bps/day  HAC t={su_s['spread_t']:+.2f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (edge=0, quiet days uninformative)")
print("  and must recover a planted 'quiet days precede drift' effect. 20 seeds, 1yr horizon.")
null_ts = []
for s_ in range(20):
    tape = data.synthetic_world(edge=0.0, seed=667 + s_)
    null_ts.append(st.synthetic_detect(tape, horizon=252)["nw_t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean Newey-West t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
tape = data.synthetic_world(edge=0.4, seed=667)
sy = st.synthetic_detect(tape, horizon=252)
print(f"  planted edge=+0.4 (seed 667): on {sy['mean_on']*100:+.2f}% vs off "
      f"{sy['mean_off']*100:+.2f}%  Welch t={sy['welch_t']:+.2f}  Newey-West t={sy['nw_t']:+.2f}")
