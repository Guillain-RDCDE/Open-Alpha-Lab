"""Reproducible headline run for Study 834 — Minimum Backtest Length (MinTRL).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic, offline, no network, no real data — a pure
research-method demonstration on a synthetic world.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from min_backtest_length import data, strategy as st  # noqa: E402

FREQ = data.TRADING_DAYS
CONF = 0.95
Z = float(__import__("scipy").stats.norm.ppf(CONF))

print("# Minimum Backtest Length (MinTRL) — how long a track record must be to trust a Sharpe")
print(f"[config] daily freq={FREQ}, conf={CONF} (Z={Z:.4f}), ann_vol={data.ANN_VOL}, "
      f"base seed={data.BASE_SEED}, fingerprint={data.config_fingerprint()}  as-of 2026-06-30")

print("\n# HEADLINE — MinTRL (years) vs annualised Sharpe (Gaussian daily, 95%)")
print("  rule of thumb: MinTRL_years ~ (Z / SR)^2")
for sr in (2.0, 1.0, 0.5, 0.25):
    exact = st.min_trl_years(sr, freq=FREQ, conf=CONF)
    rule = (Z / sr) ** 2
    print(f"  SR={sr:>4.2f}:  MinTRL = {exact:6.2f} yr   (rule of thumb {rule:6.2f} yr)")

print("\n# SKEW / KURTOSIS INFLATE THE REQUIREMENT (monthly returns, SR=1.0, 95%)")
print("  monthly is the realistic hedge-fund reporting frequency, where the moment correction bites")
mtl_g = st.min_trl_years(1.0, freq=data.MONTHS, conf=CONF, skew=0.0, kurt=3.0)
for sk, ku, tag in [(0.0, 3.0, "Gaussian"),
                    (-1.0, 4.5, "skew -1, kurt 4.5"),
                    (-2.0, 9.0, "skew -2, kurt 9 (fat left tail)")]:
    mtl = st.min_trl_years(1.0, freq=data.MONTHS, conf=CONF, skew=sk, kurt=ku)
    print(f"  {tag:<32s}: MinTRL = {mtl:5.2f} yr  (x{mtl/mtl_g:4.2f} vs Gaussian)")

print("\n# SHORT BACKTESTS CANNOT TELL SKILL FROM LUCK (worthless world, true Sharpe 0)")
for ny in (1.0, 2.0):
    lp = st.luck_prob(data, threshold_sr=1.0, n_years=ny, freq=FREQ, n_sims=4000, seed=834)
    print(f"  over {ny:.0f}yr: {lp['frac']*100:5.1f}% of worthless backtests post observed "
          f"Sharpe >= 1.0  (best seen {lp['best_sr']:.2f})  [{lp['n_sims']} sims]")
sim0 = st.simulate(data, sr_ann_true=0.0, n_years=2.0, freq=FREQ, n_sims=4000, conf=CONF, seed=834)
print(f"  observed-Sharpe dispersion at 2yr: median {sim0['median_obs_sr']:+.3f}, "
      f"sd {sim0['sd_obs_sr']:.3f}  (a driftless series routinely 'looks' skilled)")

print("\n# CALIBRATION — the PSR test is unbiased on the null (should reject ~5%)")
for ny in (1.0, 2.0, 5.0):
    s = st.simulate(data, sr_ann_true=0.0, n_years=ny, freq=FREQ, n_sims=4000, conf=CONF, seed=834)
    print(f"  {ny:>4.1f}yr: PSR(0)>=0.95 fires on {s['reject_frac']*100:5.2f}% "
          f"[Wilson {s['reject_lo']*100:.2f}-{s['reject_hi']*100:.2f}%]  (nominal 5.00%)")

print("\n# POSITIVE CONTROL — a GENUINE Sharpe-1 strategy is only confirmable past its MinTRL")
mtl1 = st.min_trl_years(1.0, freq=FREQ, conf=CONF)
pow1 = st.min_trl_for_power(1.0, freq=FREQ, conf=CONF, power=CONF)
print(f"  MinTRL(SR=1) = {mtl1:.2f} yr (observed-equals-target becomes significant); "
      f"95%-power length = {pow1:.2f} yr")
pc = st.power_curve(data, sr_ann_true=1.0, year_grid=(1.0, 2.71, 5.0, 10.82),
                    freq=FREQ, n_sims=4000, conf=CONF, seed=834)
for ny, fr, lo, hi in zip(pc["years"], pc["reject_frac"], pc["reject_lo"], pc["reject_hi"]):
    print(f"  {ny:>5.1f}yr: detected {fr*100:5.1f}% of the time  "
          f"[Wilson {lo*100:.1f}-{hi*100:.1f}%]")
