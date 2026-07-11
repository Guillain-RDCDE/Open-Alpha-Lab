"""Reproducible headline run for Study 673 — T3 (Tillson).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Reads (or fetches once) five daily total-return tapes from
``_cache/``, builds Tillson's T3(14, v=0.7) price-cross timing rule, races it net of costs
against buy-and-hold and against the SMA(14)/EMA(14) rules it claims to beat, runs the
position-shuffle permutation placebo, sweeps costs, breaks down per instrument, splits the
sample, runs a long/short variant, checks the literal "hugs price / less lag" mechanism
claim (tracking distance + a deterministic step response), sweeps the volume factor v (the
study's own robustness axis) and the T3-slope variant, and finally runs the synthetic
positive control. Network is touched only on a cache miss or with ``--fetch``.

    python examples/verify.py            # cache-first
    python examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from t3_tillson import data, strategy as st  # noqa: E402

T3_N = 14      # same nominal N convention as siblings 432/672 for a fair cross-study race
T3_V = 0.7     # Tillson's own suggested "volume factor" (Better Moving Averages, 1998)
COST_BPS = 5.0

print("# T3 (Tillson) — does the six-times-smoothed, volume-factor MA give a genuinely "
      "lower-lag, cleaner-crossover trend filter than a plain SMA/EMA?")

fetch = "--fetch" in sys.argv
if fetch or not data.have_real():
    print("(fetching the basket once)")
    data.fetch_all()

bars = {t: data.load_real(t, asof=data.AS_OF) for t in data.TICKERS}

print(f"\n# Data stamp (as-of {data.AS_OF}, auto_adjust=True total-return bars)")
for t in data.TICKERS:
    print(data_stamp(t, bars[t], cols=["close"], asof=data.AS_OF))

spy = bars["SPY"]

print("\n# THE MECHANISM — does T3 actually track price more tightly / react faster?")
td = st.tracking_distance(spy["close"], n=T3_N, v=T3_V)
print(f"  mean |close - line| / close  (SPY, N={T3_N}, v={T3_V}):  "
      f"T3={td['T3']:.3f}%  SMA={td['SMA']:.3f}%  EMA={td['EMA']:.3f}%")
sr = st.step_response(n=T3_N, v=T3_V)
after5 = sr.iloc[35]
print(f"  deterministic +20% step response (N={T3_N}, v={T3_V}): 5 bars after the jump, "
      f"price is {sr['price'].iloc[35]:.1f} — T3 has reached {after5['T3']:.2f}, "
      f"SMA {after5['SMA']:.2f}, EMA {after5['EMA']:.2f} (higher = caught up more)")

print(f"\n# THE HEADLINE — SPY, T3({T3_N}, v={T3_V}) price-cross vs SMA({T3_N}) vs EMA({T3_N}) "
      f"vs buy&hold, net of {COST_BPS:.0f} bps one-way cost")
r = st.run_experiment(spy, t3_n=T3_N, t3_v=T3_V, sma_period=T3_N, ema_period=T3_N,
                      cost_bps=COST_BPS)
for k in ("T3", "SMA", "EMA"):
    v = r[k]
    print(f"  {k:3s} Sharpe_net={v['sharpe_net']:+.3f}  CAGR={v['cagr_net']*100:+.2f}%  "
          f"MaxDD={v['maxdd_net']*100:.1f}%  spread={v['mean_spread_bps']:+.2f}bps/day  "
          f"HAC t={v['spread_t']:+.2f}  switches/yr={v['switches_per_yr']:.1f}  "
          f"time-in-mkt={v['time_in_market']*100:.0f}%")
bh = r["T3"]
print(f"  B&H Sharpe_net={bh['bh_sharpe']:+.3f}  CAGR={bh['bh_cagr']*100:+.2f}%  "
      f"MaxDD={bh['bh_maxdd']*100:.1f}%")

print("\n# \"Fewer whipsaws than a plain MA?\" — position switches/year")
for k in ("T3", "SMA", "EMA"):
    print(f"  {k:3s} {r[k]['switches_per_yr']:.1f} switches/yr")

print("\n# \"Cleaner / better than the plain MA it claims to beat?\" — head-to-head HAC t")
print(f"  T3 - SMA: {r['diff_t3_sma_bps']:+.2f} bps/day  HAC t = {r['diff_t3_sma_t']:+.2f}")
print(f"  T3 - EMA: {r['diff_t3_ema_bps']:+.2f} bps/day  HAC t = {r['diff_t3_ema_t']:+.2f}")

print("\n# Permutation placebo (T3, gross spread, 2,000 draws)")
p = r["T3_permutation"]
print(f"  observed={p['observed_spread_bps']:+.2f}bps  placebo_mean="
      f"{p['placebo_mean_bps']:+.2f}bps  p={p['p_value']:.4f}")

print(f"\n# Cost sweep (SPY, T3({T3_N}, v={T3_V}))")
for c in (0.0, 2.0, 5.0, 10.0):
    rr = st.run_experiment(spy, t3_n=T3_N, t3_v=T3_V, cost_bps=c)["T3"]
    print(f"  cost={c:4.1f}bps  Sharpe_net={rr['sharpe_net']:+.3f}  "
          f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# Per-instrument (T3({T3_N}, v={T3_V}) long/flat, cost={COST_BPS:.0f}bps)")
for t in data.TICKERS:
    rr = st.run_experiment(bars[t], t3_n=T3_N, t3_v=T3_V, sma_period=T3_N, ema_period=T3_N,
                           cost_bps=COST_BPS)
    m = rr["T3"]
    print(f"  {t:5s} T3_Sharpe={m['sharpe_net']:+.3f}  B&H={m['bh_sharpe']:+.3f}  "
          f"spread={m['mean_spread_bps']:+.2f}bps/day  HAC t={m['spread_t']:+.2f}  "
          f"T3-SMA t={rr['diff_t3_sma_t']:+.2f}  T3-EMA t={rr['diff_t3_ema_t']:+.2f}  "
          f"switches/yr T3={m['switches_per_yr']:.1f} SMA={rr['SMA']['switches_per_yr']:.1f} "
          f"EMA={rr['EMA']['switches_per_yr']:.1f}")

print(f"\n# SPY in/out-of-sample split (T3({T3_N}, v={T3_V}), cost={COST_BPS:.0f}bps)")
half = len(spy) // 2
for lbl, seg in (("H1", spy.iloc[:half]), ("H2", spy.iloc[half:])):
    rr = st.run_experiment(seg, t3_n=T3_N, t3_v=T3_V, cost_bps=COST_BPS)["T3"]
    print(f"  {lbl}: {seg.index[0].date()}->{seg.index[-1].date()}  "
          f"Sharpe_net={rr['sharpe_net']:+.3f}  B&H={rr['bh_sharpe']:+.3f}  "
          f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# Long/short variant (SPY, T3({T3_N}, v={T3_V}), cost={COST_BPS:.0f}bps, "
      "borrow 50bps/yr)")
rr = st.run_experiment(spy, t3_n=T3_N, t3_v=T3_V, cost_bps=COST_BPS, long_short=True)["T3"]
print(f"  Sharpe_net={rr['sharpe_net']:+.3f}  B&H={rr['bh_sharpe']:+.3f}  "
      f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# T3-SLOPE timer variant (SPY, N={T3_N}, v={T3_V}, cost={COST_BPS:.0f}bps) — "
      "\"T3-slope timer\" half of the brief")
rr = st.run_experiment(spy, t3_n=T3_N, t3_v=T3_V, cost_bps=COST_BPS, rule="slope")["T3"]
print(f"  Sharpe_net={rr['sharpe_net']:+.3f}  B&H={rr['bh_sharpe']:+.3f}  "
      f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}  "
      f"switches/yr={rr['switches_per_yr']:.1f}")

print(f"\n# Volume-factor (v) robustness sweep (SPY, T3 price-cross, N={T3_N}, "
      f"cost={COST_BPS:.0f}bps)")
sweep = st.v_sweep(spy, cost_bps=COST_BPS, t3_n=T3_N)
for vv, row in sweep.iterrows():
    print(f"  v={vv:.1f}  Sharpe_net={row['sharpe_net']:+.3f}  "
          f"spread={row['spread_bps']:+.2f}bps/day  HAC t={row['spread_t']:+.2f}  "
          f"switches/yr={row['switches_per_yr']:.1f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the Welch/HAC detector must NOT fire on a null world (edge=0) and must recover")
print("  a planted trend. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    b, _ = data.synthetic_panel(n_days=6000, edge=0.0, seed=673 + s_)
    null_ts.append(st.run_experiment(b, t3_n=T3_N, t3_v=T3_V, cost_bps=0.0)["T3"]["spread_t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean HAC t = {null_ts.mean():+.2f} (sd "
      f"{null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for e in (0.3, 0.6, 1.0):
    b, _ = data.synthetic_panel(n_days=6000, edge=e, seed=673)
    sy = st.run_experiment(b, t3_n=T3_N, t3_v=T3_V, cost_bps=0.0)["T3"]
    print(f"  planted edge={e:.1f} (seed 673): spread {sy['mean_spread_bps']:+.2f}bps/day  "
          f"HAC t = {sy['spread_t']:+.2f}  Sharpe_net = {sy['sharpe_net']:+.2f}")
