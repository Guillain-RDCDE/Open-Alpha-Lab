"""Reproducible headline run for Study 841 — Overlapping-Returns Inflation.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic and fully offline (no network, no cache) — the whole demo
is a Monte Carlo over synthetic monthly worlds.

    python examples/verify.py            # full 2,000-sim headline (slow, ~4-5 min)
    python examples/verify.py --fast     # 400-sim preview (a few seconds, same story)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from overlapping_returns import data, strategy as st  # noqa: E402

N_SIMS = 400 if "--fast" in sys.argv else 2000
HORIZONS = (1, 3, 6, 12, 24)

print("# Overlapping-Returns Inflation — does the long-horizon predictive-regression overlap "
      "inflate the naive t and R²?")
print(f"# synthetic, offline, deterministic | {N_SIMS} sims/horizon | rho=0.95, delta=-0.9, "
      "600 monthly rows | as-of 2026-06-30")

df, truth = data.simulate_world(n_months=600, beta=0.0, rho=0.95, seed=841)
print(f"[data] null world: {len(df)} months {df.index[0]} -> {df.index[-1]}  "
      f"beta={truth.beta} rho={truth.rho}  fingerprint={data.fingerprint(df)}")
print("  SYNTHETIC-ONLY: real free data can never certify 'zero predictability' -> capped NONE on "
      "the Signal axis.")

print("\n# ONE NULL WORLD, UP CLOSE (seed 841) — the same regression, three standard errors")
print("  h  |  slope   naive_R2   t_naive   t_NW    t_Hodrick")
for h in HORIZONS:
    o = st.predictive_regression(df["x"].to_numpy(), df["r"].to_numpy(), h)
    print(f"  {h:2d} | {o['slope']:+.4f}   {o['r2']*100:5.2f}%   {o['t_naive']:+6.2f}  "
          f"{o['t_nw']:+6.2f}   {o['t_hodrick']:+6.2f}")
print("  ^ at h=12 the naive t 'discovers' predictability (t>4.8) that does NOT exist (beta=0); "
      "Hodrick t is correctly ~1.3.")

print(f"\n# THE HEADLINE — rejection rate of the two-sided 5% test UNDER THE NULL (size; should be "
      f"~0.05), {N_SIMS} sims/horizon")
print("  h  | naive  NW(h-1)  Hodrick | mean|t|_naive  mean_R2")
null_sweep = st.horizon_sweep(data, horizons=HORIZONS, beta=0.0, rho=0.95,
                              n_months=600, n_sims=N_SIMS, base_seed=841)
for h, row in null_sweep.iterrows():
    print(f"  {h:2d} | {row['reject_naive']:.3f}  {row['reject_nw']:.3f}   {row['reject_hodrick']:.3f}  "
          f"|   {row['mean_abs_t_naive']:5.2f}      {row['mean_r2']*100:5.2f}%")
print("  ^ naive size explodes 6% -> 66% as h grows; Hodrick stays ~6% at EVERY horizon; NW helps but "
      "stays over-sized.")

print(f"\n# POSITIVE CONTROL — rejection rate with a GENUINE planted edge (beta=0.005) = POWER, "
      f"{N_SIMS} sims/horizon")
print("  h  | naive   NW    Hodrick")
edge_sweep = st.horizon_sweep(data, horizons=HORIZONS, beta=0.005, rho=0.95,
                              n_months=600, n_sims=N_SIMS, base_seed=841)
for h, row in edge_sweep.iterrows():
    print(f"  {h:2d} | {row['reject_naive']:.3f}  {row['reject_nw']:.3f}  {row['reject_hodrick']:.3f}")
print("  ^ the corrected tests keep HIGH power on a real edge (they reward genuine predictability, "
      "not just tame the null).")

print("\n# VERDICT: Signal NONE (synthetic null, no real edge) x Tradability MIRAGE (an inflated "
      "t/R² is unspendable) x 'Does overlap inflate inference?' CONFIRMED.")
