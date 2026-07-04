"""Reproducible headline run for Study 625 — Starting-Yield-Bond-Decade.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; reads the spliced GS10 tape from
``_cache/gs10_extended.csv`` (built once from the staged Shiller file +
yfinance ^TNX), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starting_yield_bond_decade import data, strategy as st

print("# Starting-Yield-Bond-Decade — Shiller GS10 1871+ spliced with ^TNX")
panel = data.load_bond_panel()
g = data.load_gs10()
n_sh = int((g["source"] == "shiller").sum())
n_tnx = int((g["source"] == "tnx").sum())
print(f"yield tape      : {len(g)} months, {g.index.min():%Y-%m} -> {g.index.max():%Y-%m} "
      f"(shiller {n_sh} months to {g[g['source']=='shiller'].index.max():%Y-%m}, "
      f"^TNX extension {n_tnx} months)")
print(f"fingerprint     : {data.fingerprint(g)}")
usable = panel.dropna(subset=["gs10", "fwd10"])
print(f"decade sample   : forward-decade defined {usable.index.min():%Y-%m} -> "
      f"{usable.index.max():%Y-%m} ({len(usable)} overlapping monthly windows)")

print("\n# PRIMARY — non-overlapping decades (unit = decade), anchor = first month")
d = st.decade_ols(panel, phase=0)
print(f"  decades       : n = {d['n']}  starts {d['starts'][0]} ... {d['starts'][-1]}")
print(f"  OLS           : R2 = {d['r2']:.3f}   slope = {d['slope']:+.3f}  "
      f"(t = {d['t_slope']:+.2f})   intercept = {d['intercept']*100:+.2f}%/yr "
      f"(t = {d['t_intercept']:+.2f})")
print(f"  identity test : t(slope=1) = {d['t_slope_vs1']:+.2f}   "
      f"MAE of the 'return = starting yield' rule = {d['mae_identity']*100:.2f} pp/yr")
print(f"  residual std  : {d['resid_std']*100:.2f} pp/yr around the fit")

print("\n# Phase sweep — all 120 monthly anchors of the non-overlapping grid")
ps = st.phase_sweep(panel)
print(f"  R2 min/med/max: {ps['r2_min']:.3f} / {ps['r2_med']:.3f} / {ps['r2_max']:.3f} "
      f"across {ps['n_phases']} phases (n = {ps['n_min']}-{ps['n_max']} decades each)")
print(f"  slope t       : min {ps['t_min']:+.2f}, median {ps['t_med']:+.2f}")

print("\n# SECONDARY — all overlapping monthly windows, Newey-West HAC (120 lags)")
h = st.overlapping_hac(panel, lags=120)
print(f"  n = {h['n']:,} windows  R2 = {h['r2']:.3f}  slope = {h['slope']:+.3f}  "
      f"HAC t = {h['hac_t_slope']:+.2f}")

print("\n# Sub-period robustness (non-overlapping decades within each half)")
half = usable.index[len(usable) // 2]
early = st.decade_ols(panel.loc[:half], phase=0)
late_panel = panel.loc[half:]
late = st.decade_ols(late_panel, phase=0)
print(f"  first half    : {early['starts'][0]}..{early['starts'][-1]}  n={early['n']}  "
      f"R2 = {early['r2']:.3f}  slope = {early['slope']:+.2f} (t = {early['t_slope']:+.2f})")
print(f"  second half   : {late['starts'][0]}..{late['starts'][-1]}  n={late['n']}  "
      f"R2 = {late['r2']:.3f}  slope = {late['slope']:+.2f} (t = {late['t_slope']:+.2f})")

print("\n# The 2020 cohort autopsy — bought the decade at 0.93% (2020-12), PARTIAL decade")
c = st.cohort_autopsy(panel, start="2020-12-01")
print(f"  start yield   : {c['start_yield']*100:.2f}%  ({c['start']})")
print(f"  realised so far ({c['years_elapsed']:.1f}y to {c['end']}): "
      f"{c['realized_ann']*100:+.2f}%/yr annualised, max drawdown {c['max_drawdown']*100:.1f}%")
print(f"  to still land the predicted {c['start_yield']*100:.2f}%/yr decade, the remaining "
      f"{10 - c['years_elapsed']:.1f} years must return {c['required_rest_ann']*100:+.2f}%/yr")

print("\n# THIRD AXIS — does the same trick work for STOCKS? (1/CAPE vs forward 10y nominal TR)")
stk = data.load_stock_decades()
sd = st.decade_ols(stk, xcol="ey", ycol="fwd10", phase=0)
print(f"  decades       : n = {sd['n']}  starts {sd['starts'][0]} ... {sd['starts'][-1]}")
print(f"  OLS           : R2 = {sd['r2']:.3f}  slope = {sd['slope']:+.2f} "
      f"(t = {sd['t_slope']:+.2f})  MAE of 'return = starting yield' = "
      f"{sd['mae_identity']*100:.2f} pp/yr")
sh_hac = st.overlapping_hac(stk, xcol="ey", ycol="fwd10", lags=120)
print(f"  overlapping   : R2 = {sh_hac['r2']:.3f}  HAC t = {sh_hac['hac_t_slope']:+.2f}  "
      f"(n = {sh_hac['n']:,})")
print(f"  bonds R2 {d['r2']:.2f} vs stocks R2 {sd['r2']:.2f} on the same design — "
      f"the arithmetic is bond-specific (the stock fix is study 120's ECY)")

print("\n# Honesty check — the lock is NOMINAL only (same regression, CPI-deflated returns)")
rp = data.load_bond_panel_real()
dr = st.decade_ols(rp, xcol="gs10", ycol="fwd10_real", phase=0)
print(f"  starting NOMINAL yield vs forward 10y REAL return: n = {dr['n']} decades  "
      f"R2 = {dr['r2']:.3f}  slope = {dr['slope']:+.2f} (t = {dr['t_slope']:+.2f})")
print(f"  nominal R2 {d['r2']:.2f} -> real R2 {dr['r2']:.2f}: the starting yield pins your "
      f"nominal decade, not your purchasing power")

print("\n# Tradability arithmetic — locking the decade")
SPREAD_BP = 2.0  # one-way on-the-run 10y note spread, conservative retail
print(f"  vehicle: on-the-run 10y note (TreasuryDirect commission $0; secondary spread "
      f"~{SPREAD_BP:.0f} bp one-way, conservative)")
print(f"  one entry per decade -> cost drag = {SPREAD_BP:.1f} bp / 10 yr = "
      f"{SPREAD_BP/10:.2f} bp/yr, vs forecast MAE {d['mae_identity']*100:.2f} pp/yr "
      f"({d['mae_identity']*1e4/ (SPREAD_BP/10):,.0f}x larger)")
print("  capacity: US Treasury market (> $25T outstanding); no borrow, long-only, no shorting")

print("\n# Synthetic control — machinery proof only, never market evidence")
print("  planted-effect world (deterministic seed 625) and seed-AVERAGED nulls (>= 20 seeds):")
for link in (1.0, 0.5, 0.0):
    r2s, tvals = [], []
    for seed in range(625, 625 + 20):
        frame, truth = data.synthetic_world(link=link, seed=seed)
        s = st.decade_ols(frame, phase=0)
        r2s.append(s["r2"])
        tvals.append(s["t_slope"])
    r2s, tvals = np.array(r2s), np.array(tvals)
    rej = float((np.abs(tvals) >= 2.0).mean())
    print(f"  link={link:.1f}: mean R2 = {r2s.mean():.3f}  mean t = {tvals.mean():+.2f}  "
          f"share |t|>=2 = {rej*100:.0f}%  (20 seeds, n=14 decades each)")
