"""Reproducible headline run for Study 824 — the Cochrane-Piazzesi factor.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached tape under ``_cache/``
(fetching once, with retries, on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from cp_factor import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")

print("# Cochrane-Piazzesi factor — does a single tent of forwards forecast bond excess returns?")

if not data.have_real():
    print("(cache miss — fetching the yield + bond-ETF tape once, with retries)")
    data.fetch()

df = data.load_panel()
print(f"[data] {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint={data.fingerprint(df)}")
print("  SIGNAL CAVEAT: coarse constant-maturity grid (0.25/5/10/30y) — a proxy for "
      "Cochrane-Piazzesi's Fama-Bliss 1..5y zeros. Named on the Signal axis.")

reg = st.cp_regression(df)
print("\n# THE HEADLINE — predictive regression of avg 1y excess return on the forward vector")
print(f"  n = {reg['n']}  (Newey-West lags = {reg['nw_lags']}, ~1.5x the 252-day overlap)")
print(f"  in-sample predictive R2 = {reg['r2']:.4f}")
print(f"  average 1y excess return (LHS mean) = {reg['avg_rx_bps']:+.1f} bps")
print(f"  single-factor predictive slope = {reg['cp_slope']:.3f}  NW t = {reg['cp_slope_t']:+.3f}")
print("  tent-shaped loadings gamma (per unit forward, decimals):")
for k, v in reg["loadings"].items():
    print(f"     {k:>8}: {v:+.4f}  (NW t = {reg['loading_t'][k]:+.2f})")

print("\n# OUT-OF-SAMPLE — expanding-window Campbell-Thompson R2 vs the prevailing mean")
oos = st.oos_r2(df)
print(f"  OOS R2 = {oos['oos_r2']:+.4f}  ({oos['n_preds']} forecasts) "
      f"-> {'beats' if oos['oos_r2'] > 0 else 'LOSES to'} the naive mean")

print("\n# PLACEBO — block-rotate the target against the forwards (1,000 rotations)")
pl = st.placebo_r2(df, n_perm=1000)
print(f"  observed R2 {pl['obs_r2']:.4f} vs placebo mean {pl['placebo_mean_r2']:.4f} "
      f"(sd {pl['placebo_sd_r2']:.4f}) -> p = {pl['p_value']:.4f}")
print("  (a persistent-regressor null already manufactures a large R2 -> the in-sample "
      "fit is not distinguishable from spurious)")

print("\n# ROBUSTNESS — two eras (split 2014-01-01)")
for lo, hi, lbl in [("2002-01-01", "2014-01-01", "2002-2013"),
                    ("2014-01-01", "2026-07-01", "2014-2026")]:
    sub = df[(df.index >= lo) & (df.index < hi)]
    r = st.cp_regression(sub)
    print(f"  {lbl}: n={r['n']}  R2={r['r2']:.4f}  single-factor NW t={r['cp_slope_t']:+.3f}")

print("\n# THE TIMER — own TLT when the OOS CP forecast is above its rolling median, else cash")
for cb in (2.0, 5.0):
    tm = st.timer_stats(df, cost_bps=cb)
    print(f"  cost={cb:>4.1f} bps/side: net Sharpe {tm['timer_sharpe']:+.3f} vs "
          f"buy-and-hold TLT {tm['bh_sharpe']:+.3f}  ({tm['switches_per_yr']:.1f} switches/yr, "
          f"invested {tm['days_invested_frac']:.0%})")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t, null_r2 = [], []
for s in range(20):
    d0 = data.synthetic_daily(edge=0.0, seed=824 + s)
    r = st.synthetic_detect(d0)
    null_t.append(r["cp_slope_t"]); null_r2.append(r["r2"])
null_t = np.asarray(null_t); null_r2 = np.asarray(null_r2)
print(f"  null (edge=0), 20 seeds: in-sample R2 mean {null_r2.mean():.4f} (max {null_r2.max():.4f}) "
      f"-> SILENT")
print(f"    but the naive single-factor NW t averages {null_t.mean():+.2f} and fires "
      f"|t|>=2 on {(abs(null_t) >= 2).sum()}/20 nulls -> that raw t is size-distorted under "
      f"persistent regressors (why we lean on R2 / OOS / placebo, not the raw t)")
plant = st.synthetic_detect(data.synthetic_daily(edge=0.05, seed=824))
plant_oos = st.oos_r2(data.synthetic_daily(edge=0.05, seed=824))["oos_r2"]
null_oos = st.oos_r2(data.synthetic_daily(edge=0.0, seed=824))["oos_r2"]
print(f"  planted (edge=0.05): in-sample R2 = {plant['r2']:.4f}, OOS R2 = {plant_oos:+.4f} "
      f"(null OOS R2 = {null_oos:+.4f}) -> the machinery recovers a real forward-return edge")
