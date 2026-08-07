"""Reproducible headline run for Study 839 — The t > 3 Threshold.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Fully deterministic and offline — a synthetic factor-zoo
method demo, no network, no real market data.

    python studies/839-tstat-three-threshold/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from tstat_threshold import data, strategy as st  # noqa: E402

ALPHA = 0.05
N_FACTORS = 1000
N_PERIODS = 240
N_TRUE = 50
EXP_T = 4.0
HLZ_N = 316  # the number of factors HLZ (2016) tally in the published zoo

print("# The t > 3 Threshold — Harvey, Liu & Zhu (2016)")
print(f"# synthetic factor zoo: N={N_FACTORS} factors x T={N_PERIODS} periods, alpha={ALPHA}")

# --- the pure-null headline zoo ------------------------------------------------
R0, tr0_mask, tr0 = data.synthetic_zoo(
    n_factors=N_FACTORS, n_periods=N_PERIODS, n_true=0, seed=839
)
print(f"\n[data] pure-null zoo  fingerprint={data.fingerprint(R0)}  "
      f"has_edge={tr0.has_edge}  (as-of 2026-06-30)")
z0 = st.zoo_stats(R0)
print(f"  cleared t>2: {z0['n_gt2']}/{N_FACTORS} ({z0['frac_gt2']*100:.2f}%)  "
      f"| cleared t>3: {z0['n_gt3']}/{N_FACTORS} ({z0['frac_gt3']*100:.2f}%)  "
      f"| max |t| = {z0['max_t']:.2f}")
print(f"  theory: t>2 -> {st.prob_exceed(2.0)*100:.2f}%   t>3 -> {st.prob_exceed(3.0)*100:.3f}%   "
      f"ratio ~{st.prob_exceed(2.0)/st.prob_exceed(3.0):.1f}x")

print("\n# THRESHOLD SUMMARY — observed vs null-expected (pure-null zoo)")
print(st.threshold_summary(st.factor_tstats(R0), thresholds=(2.0, 3.0)).round(4).to_string())

# --- the multiple-testing hurdles (HLZ's N=316) --------------------------------
print(f"\n# MULTIPLE-TESTING HURDLES — how high the bar must rise (N={HLZ_N}, alpha={ALPHA})")
print(f"  Bonferroni |t| cutoff (N={HLZ_N}): {st.bonferroni_t(HLZ_N, ALPHA):.2f}")
for n in (1, 10, 100, HLZ_N, 1000):
    print(f"    Bonferroni |t|(N={n:>4}) = {st.bonferroni_t(n, ALPHA):.2f}")

# --- the full correction table on the null zoo ---------------------------------
print(f"\n# CORRECTION TABLE — implied |t| cutoff & discoveries on the pure-null zoo (N={N_FACTORS})")
mt0 = st.multiple_testing_table(st.factor_tstats(R0), alpha=ALPHA)
print(mt0.round(3).to_string())

# --- the planted mixture: FDR collapse t>2 -> t>3 ------------------------------
Rm, is_true, trm = data.synthetic_zoo(
    n_factors=N_FACTORS, n_periods=N_PERIODS, n_true=N_TRUE, expected_t=EXP_T, seed=839
)
print(f"\n# PLANTED MIXTURE — {N_TRUE} true (expected |t|={EXP_T}) buried in {N_FACTORS} factors  "
      f"fingerprint={data.fingerprint(Rm)}")
zm = st.zoo_stats(Rm, is_true)
d2, d3 = zm["det2"], zm["det3"]
print(f"  t>2: {d2['n_disc']} discoveries = {d2['tp']} true + {d2['fp']} false  "
      f"-> FDR {d2['fdr']*100:.1f}%  power {d2['power']*100:.1f}%")
print(f"  t>3: {d3['n_disc']} discoveries = {d3['tp']} true + {d3['fp']} false  "
      f"-> FDR {d3['fdr']*100:.1f}%  power {d3['power']*100:.1f}%")
print(f"  correction table on the mixture:")
print(st.multiple_testing_table(st.factor_tstats(Rm), alpha=ALPHA).round(3).to_string())

# --- publication haircut -------------------------------------------------------
print(f"\n# PUBLICATION HAIRCUT — a claimed |t| once a {HLZ_N}-test search is disclosed (Bonferroni)")
for tt in (2.0, 2.5, 3.0, 3.5, 4.0):
    h = st.publication_haircut(tt, HLZ_N)
    print(f"  reported |t|={tt:.1f}: p={h['p_naive']:.2e} -> p_adj={h['p_adjusted']:.3f}  "
          f"eff |t|={h['t_adjusted']:.2f}  haircut {h['haircut']*100:.0f}%  "
          f"survives0.05={h['survives_005']}")

# --- seed-robust controls (>= 20 seeds) ----------------------------------------
print("\n# SEED-ROBUST NULL CONTROL (20 seeds) — the fractions are unbiased")
sn = st.seed_robust_null(data, n_factors=N_FACTORS, n_periods=N_PERIODS, n_seeds=20)
print(f"  mean frac t>2 = {sn['mean_frac_gt2']*100:.2f}% (theory {sn['theory_frac_gt2']*100:.2f}%)  "
      f"| mean frac t>3 = {sn['mean_frac_gt3']*100:.3f}% (theory {sn['theory_frac_gt3']*100:.3f}%)")
print(f"  mean n t>2 = {sn['mean_n_gt2']:.1f}  | mean n t>3 = {sn['mean_n_gt3']:.1f}  "
      f"| ratio = {sn['ratio_gt2_over_gt3']:.1f}x  | mean max|t| = {sn['mean_max_t']:.2f}")

print("\n# SEED-ROBUST MIXTURE CONTROL (20 seeds) — FDR collapses t>2 -> t>3, BHY keeps the real ones")
sm = st.seed_robust_mixture(
    data, n_factors=N_FACTORS, n_true=N_TRUE, expected_t=EXP_T,
    n_periods=N_PERIODS, n_seeds=20,
)
print(f"  FDR: t>2 {sm['mean_fdr_t2']*100:.1f}%  ->  t>3 {sm['mean_fdr_t3']*100:.1f}%  "
      f"(collapse {sm['fdr_collapse']:.1f}x)")
print(f"  power: t>2 {sm['mean_power_t2']*100:.1f}%  ->  t>3 {sm['mean_power_t3']*100:.1f}%")
print(f"  BHY: {sm['mean_bhy_n']:.1f} discoveries, implied |t| cutoff {sm['mean_bhy_cutoff']:.2f}, "
      f"realized FDR {sm['mean_bhy_fdr']*100:.1f}%")

print("\n# VERDICT: Signal NONE (synthetic-only; nothing real to find) x "
      "Tradability MIRAGE x 'Does t>2 inflate false discoveries?' CONFIRMED")
