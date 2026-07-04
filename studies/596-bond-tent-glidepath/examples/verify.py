"""Reproducible headline run for Study 596 — Bond Tent (Rising Equity Glidepath).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the study's own
``_cache/shiller_sp500.parquet`` (Shiller 1871+); the synthetic control always
runs offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_tent_glidepath import data, strategy as st

N_BOOT = 1000
WR_GRID = (0.040, 0.045, 0.050, 0.055)
PAIRS = (("rising", "s6040"), ("rising", "s5050"),
         ("rising", "declining"), ("s5050", "s6040"))
PAIR_LABEL = {
    ("rising", "s6040"): "HEADLINE  rising - 60/40 (the Kitces-Pfau claim)",
    ("rising", "s5050"): "SHAPE     rising - 50/50 (matched 50% avg equity)",
    ("rising", "declining"): "TIMING    rising - declining (same avg, mirrored)",
    ("s5050", "s6040"): "ALLOCATION 50/50 - 60/40 (just holding less equity)",
}

print("# Bond Tent (Rising Equity Glidepath) -- Shiller real-return tape")
df = data.real_returns()
eq, bd = df["EQ"].to_numpy(), df["BD"].to_numpy()
ge, gb, starts = st.cohort_year_returns(eq, bd)
dates = df.index[starts]
ann_eq = 100 * ((1 + df["EQ"]).prod() ** (12 / len(df)) - 1)
ann_bd = 100 * ((1 + df["BD"]).prod() ** (12 / len(df)) - 1)
print(f"tape        : {df.index.min().date()} -> {df.index.max().date()} "
      f"({len(df)} months)  fingerprint {data.fingerprint(df)}")
print(f"real returns: EQ {ann_eq:.2f}%/yr  BD {ann_bd:.2f}%/yr "
      f"(10y approx, CPI-deflated)  realised premium {ann_eq - ann_bd:.2f} pp")
print(f"cohorts     : {ge.shape[0]} monthly-start 30-year retirements, "
      f"{dates.min().date()} -> {dates.max().date()};  cost "
      f"{st.COST_BPS:.0f} bps one-way x traded value")

print("\n# Success rates (share of cohorts never depleted, 30y, real withdrawals)")
res_by_wr = {wr: st.run_all(ge, gb, wr) for wr in WR_GRID}
hdr = "  wr      " + "".join(f"{s:>11}" for s in st.STRATEGIES)
print(hdr)
for wr in WR_GRID:
    row = "".join(f"{100*res_by_wr[wr][s]['success'].mean():>10.2f}%"
                  for s in st.STRATEGIES)
    print(f"  {100*wr:.1f}%  {row}")

print("\n# Terminal real wealth at the 4% rule (multiple of initial)")
res4, res5 = res_by_wr[0.040], res_by_wr[0.050]
for s in st.STRATEGIES:
    m = st.summarize(res4[s])
    print(f"  {s:<10} mean {m['mean']:.3f}  median {m['median']:.3f}  "
          f"p05 {m['p05']:.3f}  worst {m['worst']:.3f}  "
          f"failures {m['n_fail']}/{m['n_cohorts']}")

print("\n# The decomposition -- HAC t (Newey-West, bandwidth = full 360-month overlap)")
print("#   on per-cohort terminal-wealth differences at 4%; success-rate gaps at 4% and 5%")
for p in PAIRS:
    a, b = p
    d = res4[a]["terminal"] - res4[b]["terminal"]
    t = st.hac_tstat(d, lags=360)
    ds4 = 100 * (res4[a]["success"].mean() - res4[b]["success"].mean())
    ds5 = 100 * (res5[a]["success"].mean() - res5[b]["success"].mean())
    print(f"  {PAIR_LABEL[p]}")
    print(f"      mean dTW@4% {d.mean():+.4f}  HAC t = {t:+.2f}   "
          f"dsuccess: {ds4:+.2f} pp @4%, {ds5:+.2f} pp @5%")

print(f"\n# Circular block bootstrap ({N_BOOT} reps, 120-month blocks, seed 596)")
print("#   NOTE: blocks preserve within-decade dynamics but destroy cross-decade")
print("#   mean reversion -- stated, not hidden. 95% percentile CIs.")
boot = st.block_bootstrap(eq, bd, pairs=PAIRS, n_boot=N_BOOT, seed=596)
for p in PAIRS:
    lo_m, hi_m = st.ci(boot[p]["dmean"])
    lo_s, hi_s = st.ci(boot[p]["dsucc"])
    print(f"  {p[0]:>7} - {p[1]:<9} dmeanTW@4% CI [{lo_m:+.3f}, {hi_m:+.3f}]"
          f"   dsuccess@5% CI [{lo_s:+.2f}, {hi_s:+.2f}] pp")

print("\n# High-CAPE retirements (top-quartile PE10 at retirement -- the tent's")
print("#   advertised sweet spot). Success rates within those cohorts:")
raw = data.load_shiller()
cape = raw["PE10"].reindex(dates).to_numpy()
valid = cape > 0
q4 = valid & (cape >= np.nanquantile(cape[valid], 0.75))
print(f"  cohorts with CAPE: {int(valid.sum())}; top-quartile n = {int(q4.sum())}")
for wr in (0.040, 0.045, 0.050):
    r = res_by_wr[wr]
    row = "".join(f"{100*r[s]['success'][q4].mean():>10.1f}%" for s in st.STRATEGIES)
    print(f"  {100*wr:.1f}%  {row}")

print("\n# Famous cohorts -- terminal real wealth at 4% (multiple of initial)")
for y in ("1929-09-01", "1966-01-01", "1972-12-01"):
    i = np.where(dates == pd.Timestamp(y))[0][0]
    row = "  ".join(f"{s} {res4[s]['terminal'][i]:.3f}" for s in st.STRATEGIES)
    print(f"  retire {y[:7]}: {row}")

f6040 = ~res4["s6040"]["success"]
fy = sorted(set(d.year for d in dates[f6040]))
print(f"\n# Rescue test -- the {int(f6040.sum())} cohorts that FAIL 60/40 at 4% "
      f"(all start {fy[0]}-{fy[-1]}):")
for s in ("rising", "declining", "s5050", "s4060"):
    r = 100 * res4[s]["success"][f6040].mean()
    print(f"  {s:<10} rescues {r:.1f}% of them")

print("\n# SAFEMAX -- highest real withdrawal rate surviving EVERY cohort")
for s in st.STRATEGIES:
    print(f"  {s:<10} {100*st.safemax(ge, gb, s):.2f}%")

print("\n# Robustness -- tent shape variants (all vs static 60/40)")
for v in ("rising", "rising_3060", "rising_fast", "rising_2080"):
    w = st.weights_for(v)
    r4 = st.simulate(ge, gb, w, wr=0.040)
    r5 = st.simulate(ge, gb, w, wr=0.050)
    t = st.hac_tstat(r4["terminal"] - res4["s6040"]["terminal"], lags=360)
    print(f"  {v:<12} succ@4% {100*r4['success'].mean():.2f}%  "
          f"succ@5% {100*r5['success'].mean():.2f}%  "
          f"meanTW@4% {r4['terminal'].mean():.3f}  HAC t vs 60/40 = {t:+.2f}")

print("\n# Robustness -- one-way cost level (success @4%)")
for cb in (0.0, 10.0, 25.0):
    ra = st.run_all(ge, gb, 0.040, cost_bps=cb)
    row = "  ".join(f"{s} {100*ra[s]['success'].mean():.2f}%"
                    for s in ("s6040", "rising", "declining"))
    print(f"  cost {cb:>4.0f} bps: {row}")

print("\n# Synthetic control -- 20 independent seeded worlds per setting, offline")
print("#   detector: rising vs declining (same 50% average equity, pure timing);")
print("#   one-sample t across worlds. Machinery proof only -- never market evidence.")
rows = [
    ("EXACT NULL  wr=0, iid, premium 4.5pp", dict(reversion=0.0, wr=0.0,
                                                  eq_mu=0.065, bd_mu=0.02)),
    ("TENT WORLD  wr=4%, iid, premium 0pp", dict(reversion=0.0, wr=0.04,
                                                 eq_mu=0.02, bd_mu=0.02)),
    ("premium 2pp, wr=4%", dict(reversion=0.0, wr=0.04, eq_mu=0.04, bd_mu=0.02)),
    ("premium 4.5pp (historical), wr=4%", dict(reversion=0.0, wr=0.04,
                                               eq_mu=0.065, bd_mu=0.02)),
    ("premium 4.5pp (historical), wr=5%", dict(reversion=0.0, wr=0.05,
                                               eq_mu=0.065, bd_mu=0.02)),
]
for label, kw in rows:
    r = st.control_shape_effect(n_seeds=20, **kw)
    tsucc = "  nan" if np.isnan(r["dsucc_t"]) else f"{r['dsucc_t']:+5.2f}"
    print(f"  {label:<38} dsuccess {r['dsucc_mean']:+.2f} pp (t {tsucc})   "
          f"dp05 {r['dp05_mean']:+.4f} (t {r['dp05_t']:+.2f})"
          if not np.isnan(r["dp05_t"]) else
          f"  {label:<38} dsuccess {r['dsucc_mean']:+.2f} pp (t {tsucc})   "
          f"dp05 {r['dp05_mean']:+.4f} (t   nan)")
print("\n(the null reads ~0; the tent's benefit exists in a low-premium world at a")
print(" moderate withdrawal rate and dies -- then reverses -- as the equity premium")
print(" approaches the 4.5 pp the US tape actually paid)")
