"""Reproducible headline run for Study 835 - Spurious Regression (Granger & Newbold 1974).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Fully deterministic, offline, synthetic-only (no network,
no real data). Run:

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from spurious_regression import data, strategy as st  # noqa: E402

# --- the frozen simulation config (its fingerprint stamps the run) ---------- #
CFG = dict(
    base_seed=835, n_pairs=5000, n_obs=250, sigma=1.0, drift=0.0,
    sweep_pairs=4000, sweep_grid=(50, 125, 250, 500, 1000),
    coint_pairs=300, coint_beta=1.0, coint_noise=1.0,
    timer_pairs=3000, timer_window=60, timer_entry_z=1.0,
)
FP = data.fingerprint(CFG)

print("# Spurious Regression - do two INDEPENDENT random walks fake a significant relation?")
print(f"[config] base_seed={CFG['base_seed']}  n_pairs={CFG['n_pairs']}  n_obs={CFG['n_obs']}"
      f"  as-of {data.AS_OF}  fingerprint {FP}")
print("  SYNTHETIC-ONLY method demo: the two series are built INDEPENDENT, so any "
      "significance is spurious by construction. No real tape -> capped at NONE on Signal.")

# --------------------------------------------------------------------------- #
# 1. The pitfall - level OLS on two independent random walks
# --------------------------------------------------------------------------- #
X, Y = data.independent_walks(CFG["n_pairs"], n_obs=CFG["n_obs"], seed=CFG["base_seed"])
ex = st.regression_experiment(X, Y)
lv, df = ex["level"], ex["diff"]
k = int(round(lv["reject_rate"] * CFG["n_pairs"]))
wlo, whi = st.wilson_interval(k, CFG["n_pairs"])
print("\n# 1. THE PITFALL - regress y on x in LEVELS (two independent random walks)")
print(f"  reject |t|>1.96 : {lv['reject_rate']:.4f}  (nominal 5%; Wilson 95% CI "
      f"[{wlo:.4f}, {whi:.4f}])  -> {lv['reject_rate']/0.05:.1f}x oversized")
print(f"  mean |t|        : {lv['mean_abs_t']:.2f}   median |t| {lv['median_abs_t']:.2f}")
print(f"  mean R2         : {lv['mean_r2']:.4f}   share R2>0.25 {lv['share_r2_gt_25']:.4f}")

print("\n# THE FIX #1 - first-difference: regress dy on dx (same pairs)")
print(f"  reject |t|>1.96 : {df['reject_rate']:.4f}  (back to ~5%, correctly sized)")
print(f"  mean |t|        : {df['mean_abs_t']:.2f}   mean R2 {df['mean_r2']:.4f}")

# --------------------------------------------------------------------------- #
# 2. Trending (drift) makes it far worse - the claim in one line
# --------------------------------------------------------------------------- #
Xd, Yd = data.independent_walks(CFG["n_pairs"], n_obs=CFG["n_obs"], drift=0.15,
                                seed=CFG["base_seed"])
exd = st.regression_experiment(Xd, Yd)["level"]
print("\n# 2. TRENDING SERIES (add a common deterministic drift) - even worse")
print(f"  reject |t|>1.96 : {exd['reject_rate']:.4f}   mean |t| {exd['mean_abs_t']:.2f}"
      f"   mean R2 {exd['mean_r2']:.4f}   share R2>0.25 {exd['share_r2_gt_25']:.4f}")

# --------------------------------------------------------------------------- #
# 3. Sample size makes the LEVEL test worse; the DIFF test stays correct
# --------------------------------------------------------------------------- #
print("\n# 3. MORE DATA MAKES IT WORSE - level reject rate rises with n (t ~ sqrt(T))")
print("  n_obs | level_reject  mean|t|  meanR2 | diff_reject")
for row in st.sample_size_sweep(data, n_obs_grid=CFG["sweep_grid"],
                                n_pairs=CFG["sweep_pairs"], seed_base=CFG["base_seed"]):
    print(f"  {row['n_obs']:>5} | {row['level_reject']:.4f}       {row['level_mean_abs_t']:>5.2f}"
          f"   {row['level_mean_r2']:.3f} | {row['diff_reject']:.4f}")

# --------------------------------------------------------------------------- #
# 4. Specificity - same OLS on two STATIONARY series is correctly sized
# --------------------------------------------------------------------------- #
sc = st.size_control(data, n_pairs=CFG["n_pairs"], n_obs=CFG["n_obs"], phi=0.0,
                     seed=CFG["base_seed"])
print("\n# 4. SPECIFICITY CONTROL - level OLS on two INDEPENDENT STATIONARY series")
print(f"  reject |t|>1.96 : {sc['reject_rate']:.4f}  (~5% -> the pitfall is nonstationarity, "
      f"not OLS)   mean |t| {sc['mean_abs_t']:.2f}  mean R2 {sc['mean_r2']:.4f}")

# --------------------------------------------------------------------------- #
# 5. The other fix - cointegration test tells spurious from genuine
# --------------------------------------------------------------------------- #
Xi, Yi = data.independent_walks(CFG["coint_pairs"], n_obs=CFG["n_obs"], seed=CFG["base_seed"])
ci = st.cointegration_reject_rate(Xi, Yi)
Xc, Yc = data.cointegrated_pairs(CFG["coint_pairs"], n_obs=CFG["n_obs"],
                                 beta=CFG["coint_beta"], noise_sd=CFG["coint_noise"],
                                 seed=CFG["base_seed"])
cc = st.cointegration_reject_rate(Xc, Yc)
print("\n# 5. THE FIX #2 - Engle-Granger cointegration test (positive control)")
print(f"  independent walks : reject no-coint {ci['reject_rate']:.4f}  (median p "
      f"{ci['median_pvalue']:.3f})  -> correctly finds NOTHING")
print(f"  cointegrated pair : reject no-coint {cc['reject_rate']:.4f}  (median p "
      f"{cc['median_pvalue']:.3f})  -> correctly finds the REAL relation")

# --------------------------------------------------------------------------- #
# 6. Tradability - a costed pairs trade on the spurious spread
# --------------------------------------------------------------------------- #
Xt, Yt = data.independent_walks(CFG["timer_pairs"], n_obs=CFG["n_obs"], seed=CFG["base_seed"])
print("\n# 6. TRADABILITY - mean-reversion pairs trade on the spurious spread (no look-ahead)")
for cb in (0.0, 1.0, 5.0):
    tm = st.pairs_timer(Xt, Yt, window=CFG["timer_window"], entry_z=CFG["timer_entry_z"],
                        cost_bps=cb, borrow_bps_yr=(0.0 if cb == 0 else 50.0), sigma_hint=1.0)
    print(f"  cost={cb:>4.1f} bps: gross {tm['gross_bps']:+.2f} -> net {tm['net_bps']:+.2f} "
          f"(t_net {tm['t_net']:+.2f}, Sharpe {tm['sharpe_net']:+.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")
print("  gross edge is indistinguishable from zero (|t_net|<2): the spread is a random "
      "walk, not mean-reverting -> nothing to harvest, costs only add insult.")
