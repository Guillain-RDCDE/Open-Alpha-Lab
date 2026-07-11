"""Reproducible headline run for Study 674 — VIDYA.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Reads (or fetches once) five daily total-return tapes from
``_cache/``, checks the "speeds up in trending/volatile regimes" mechanism directly
(regime correlations + a deterministic step response + tracking distance), builds the
VIDYA(14, cmo=9) price-cross timing rule, races it net of costs against buy-and-hold and
against the SMA(14)/EMA(14) rules it claims to beat, runs the position-shuffle
permutation placebo, sweeps costs, breaks down per instrument, splits the sample, runs a
long/short variant, sweeps the CMO lookback period, and finally runs the synthetic
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

from vidya import data, strategy as st  # noqa: E402

PERIOD = 14       # the nominal EMA-equivalent smoothing period (matches SMA/EMA comparators)
CMO_PERIOD = 9     # Chande's own CMO lookback (a free knob distinct from PERIOD)
COST_BPS = 5.0

print("# VIDYA — does Chande's volatility-scaled smoothing constant actually speed up "
      "in trending/volatile regimes, and does it beat a plain SMA/EMA?")

fetch = "--fetch" in sys.argv
if fetch or not data.have_real():
    print("(fetching the basket once)")
    data.fetch_all()

bars = {t: data.load_real(t, asof=data.AS_OF) for t in data.TICKERS}

print(f"\n# Data stamp (as-of {data.AS_OF}, auto_adjust=True total-return bars)")
for t in data.TICKERS:
    print(data_stamp(t, bars[t], cols=["close"], asof=data.AS_OF))

spy = bars["SPY"]

print("\n# THE MECHANISM — does VI = |CMO|/100 (VIDYA's speed knob) track volatility, "
      "or trend, or neither?")
rc = st.regime_correlations(spy["close"], cmo_period=CMO_PERIOD)
print(f"  corr(VI, realized 20d vol)   = {rc['corr_vi_vol']:+.3f}   (n={rc['n']})")
print(f"  corr(VI, 20d trend strength) = {rc['corr_vi_trend']:+.3f}")
print(f"  mean VI in low-vol tercile   = {rc['vi_low_vol_tercile']:.3f}   "
      f"high-vol tercile = {rc['vi_high_vol_tercile']:.3f}")
print(f"  mean VI in low-trend tercile = {rc['vi_low_trend_tercile']:.3f}   "
      f"high-trend tercile = {rc['vi_high_trend_tercile']:.3f}")

td = st.tracking_distance(spy["close"], period=PERIOD, cmo_period=CMO_PERIOD)
print(f"\n  mean |close - line| / close  (SPY, N={PERIOD}):  "
      f"VIDYA={td['VIDYA']:.3f}%  SMA={td['SMA']:.3f}%  EMA={td['EMA']:.3f}%")
sr = st.step_response(period=PERIOD, cmo_period=CMO_PERIOD)
after5 = sr.iloc[35]
print(f"  deterministic +20% step response (N={PERIOD}): 5 bars after the jump, price is "
      f"{sr['price'].iloc[35]:.1f} — VIDYA has reached {after5['VIDYA']:.2f}, "
      f"SMA {after5['SMA']:.2f}, EMA {after5['EMA']:.2f} (higher = caught up more)")

print(f"\n# THE HEADLINE — SPY, VIDYA({PERIOD}, cmo={CMO_PERIOD}) vs SMA({PERIOD}) vs "
      f"EMA({PERIOD}) vs buy&hold, net of {COST_BPS:.0f} bps one-way cost")
r = st.run_experiment(spy, period=PERIOD, cmo_period=CMO_PERIOD, sma_period=PERIOD,
                      ema_period=PERIOD, cost_bps=COST_BPS)
for k in ("VIDYA", "SMA", "EMA"):
    v = r[k]
    print(f"  {k:5s} Sharpe_net={v['sharpe_net']:+.3f}  CAGR={v['cagr_net']*100:+.2f}%  "
          f"MaxDD={v['maxdd_net']*100:.1f}%  spread={v['mean_spread_bps']:+.2f}bps/day  "
          f"HAC t={v['spread_t']:+.2f}  switches/yr={v['switches_per_yr']:.1f}  "
          f"time-in-mkt={v['time_in_market']*100:.0f}%")
bh = r["VIDYA"]
print(f"  B&H   Sharpe_net={bh['bh_sharpe']:+.3f}  CAGR={bh['bh_cagr']*100:+.2f}%  "
      f"MaxDD={bh['bh_maxdd']*100:.1f}%")

print("\n# \"Fewer whipsaws than a plain MA?\" — position switches/year")
for k in ("VIDYA", "SMA", "EMA"):
    print(f"  {k:5s} {r[k]['switches_per_yr']:.1f} switches/yr")

print("\n# \"Beats the plain MA it claims to beat?\" — head-to-head HAC t")
print(f"  VIDYA - SMA: {r['diff_vidya_sma_bps']:+.2f} bps/day  HAC t = {r['diff_vidya_sma_t']:+.2f}")
print(f"  VIDYA - EMA: {r['diff_vidya_ema_bps']:+.2f} bps/day  HAC t = {r['diff_vidya_ema_t']:+.2f}")

print("\n# Permutation placebo (VIDYA, gross spread, 2,000 draws)")
p = r["VIDYA_permutation"]
print(f"  observed={p['observed_spread_bps']:+.2f}bps  placebo_mean="
      f"{p['placebo_mean_bps']:+.2f}bps  p={p['p_value']:.4f}")

print(f"\n# Cost sweep (SPY, VIDYA({PERIOD}, cmo={CMO_PERIOD}))")
for c in (0.0, 2.0, 5.0, 10.0):
    rr = st.run_experiment(spy, period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=c)["VIDYA"]
    print(f"  cost={c:4.1f}bps  Sharpe_net={rr['sharpe_net']:+.3f}  "
          f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# Per-instrument (VIDYA({PERIOD}, cmo={CMO_PERIOD}) long/flat, cost={COST_BPS:.0f}bps)")
for t in data.TICKERS:
    rr = st.run_experiment(bars[t], period=PERIOD, cmo_period=CMO_PERIOD, sma_period=PERIOD,
                           ema_period=PERIOD, cost_bps=COST_BPS)
    m = rr["VIDYA"]
    print(f"  {t:5s} VIDYA_Sharpe={m['sharpe_net']:+.3f}  B&H={m['bh_sharpe']:+.3f}  "
          f"spread={m['mean_spread_bps']:+.2f}bps/day  HAC t={m['spread_t']:+.2f}  "
          f"V-SMA t={rr['diff_vidya_sma_t']:+.2f}  V-EMA t={rr['diff_vidya_ema_t']:+.2f}  "
          f"switches/yr VIDYA={m['switches_per_yr']:.1f} SMA={rr['SMA']['switches_per_yr']:.1f} "
          f"EMA={rr['EMA']['switches_per_yr']:.1f}")

print(f"\n# SPY in/out-of-sample split (VIDYA({PERIOD}, cmo={CMO_PERIOD}), cost={COST_BPS:.0f}bps)")
half = len(spy) // 2
for lbl, seg in (("H1", spy.iloc[:half]), ("H2", spy.iloc[half:])):
    rr = st.run_experiment(seg, period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=COST_BPS)["VIDYA"]
    print(f"  {lbl}: {seg.index[0].date()}->{seg.index[-1].date()}  "
          f"Sharpe_net={rr['sharpe_net']:+.3f}  B&H={rr['bh_sharpe']:+.3f}  "
          f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# Long/short variant (SPY, VIDYA({PERIOD}, cmo={CMO_PERIOD}), cost={COST_BPS:.0f}bps, "
      f"borrow 50bps/yr)")
rr = st.run_experiment(spy, period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=COST_BPS,
                       long_short=True)["VIDYA"]
print(f"  Sharpe_net={rr['sharpe_net']:+.3f}  B&H={rr['bh_sharpe']:+.3f}  "
      f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}")

print(f"\n# CMO-period robustness sweep (SPY, VIDYA period={PERIOD} fixed, cost={COST_BPS:.0f}bps)")
for cm in (5, 9, 14, 20, 30):
    rr = st.run_experiment(spy, period=PERIOD, cmo_period=cm, cost_bps=COST_BPS)["VIDYA"]
    print(f"  cmo_period={cm:2d}  Sharpe_net={rr['sharpe_net']:+.3f}  "
          f"spread={rr['mean_spread_bps']:+.2f}bps/day  HAC t={rr['spread_t']:+.2f}  "
          f"switches/yr={rr['switches_per_yr']:.1f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the HAC detector must NOT fire on a null world (edge=0) and must recover")
print("  a planted trend. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    b, _ = data.synthetic_panel(n_days=6000, edge=0.0, seed=674 + s_)
    null_ts.append(st.run_experiment(b, period=PERIOD, cmo_period=CMO_PERIOD,
                                     cost_bps=0.0)["VIDYA"]["spread_t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean HAC t = {null_ts.mean():+.2f} (sd "
      f"{null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for e in (0.3, 0.6, 1.0):
    b, _ = data.synthetic_panel(n_days=6000, edge=e, seed=674)
    sy = st.run_experiment(b, period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=0.0)["VIDYA"]
    print(f"  planted edge={e:.1f} (seed 674): spread {sy['mean_spread_bps']:+.2f}bps/day  "
          f"HAC t = {sy['spread_t']:+.2f}  Sharpe_net = {sy['sharpe_net']:+.2f}")
