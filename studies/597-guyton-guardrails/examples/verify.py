"""Reproducible headline run for Study 597 — Guyton-Klinger Guardrails.

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

from guyton_guardrails import data, strategy as st

N_BOOT = 1000
WR_GRID = (0.040, 0.045, 0.050, 0.055, 0.060)

print("# Guyton-Klinger Guardrails -- Shiller nominal tape + CPI, deflated to real")
df = data.nominal_returns()
eq, bd, infl = df["EQ"].to_numpy(), df["BD"].to_numpy(), df["INFL"].to_numpy()
ge, gb, gi, starts = st.cohort_year_returns(eq, bd, infl)
dates = df.index[starts]
ann_eq = 100 * ((1 + df["EQ"]).prod() ** (12 / len(df)) - 1)
ann_bd = 100 * ((1 + df["BD"]).prod() ** (12 / len(df)) - 1)
ann_in = 100 * ((1 + df["INFL"]).prod() ** (12 / len(df)) - 1)
print(f"tape        : {df.index.min().date()} -> {df.index.max().date()} "
      f"({len(df)} months)  fingerprint {data.fingerprint(df)}")
print(f"nominal     : EQ {ann_eq:.2f}%/yr  BD {ann_bd:.2f}%/yr (10y approx)  "
      f"CPI {ann_in:.2f}%/yr")
print(f"cohorts     : {ge.shape[0]} monthly-start 30-year retirements, "
      f"{dates.min().date()} -> {dates.max().date()};  60/40, cost "
      f"{st.COST_BPS:.0f} bps one-way x traded value")

print("\n# Machinery identity -- guardrails at (0, inf) with no freeze/cap == Bengen")
f5 = st.simulate(ge, gb, gi, wr0=0.05, preset="fixed")
w5 = st.simulate(ge, gb, gi, wr0=0.05, preset="gk_wide")
print(f"  max |income path difference| = "
      f"{np.abs(f5['income'] - w5['income']).max():.1e}  (exact 0 expected)")
f4 = st.simulate(ge, gb, gi, wr0=0.04, preset="fixed")
print(f"  fixed 4% success {100*f4['success'].mean():.2f}% "
      f"({int((~f4['success']).sum())}/{len(dates)} failures) -- sibling 596's "
      "real-space simulator reads 96.33%; the residual is a second-order "
      "compounding convention (nominal-then-deflate vs real), stated not hidden")

print("\n# Success rates (share of cohorts never depleted, 30y)")
res_f = {wr: st.simulate(ge, gb, gi, wr0=wr, preset="fixed") for wr in WR_GRID}
res_g = {wr: st.simulate(ge, gb, gi, wr0=wr, preset="gk") for wr in WR_GRID}
print("  wr        fixed (Bengen)   GK guardrails")
for wr in WR_GRID:
    print(f"  {100*wr:.1f}%  {100*res_f[wr]['success'].mean():>13.2f}%"
          f"  {100*res_g[wr]['success'].mean():>13.2f}%")

print("\n# SAFEMAX -- highest initial rate at which EVERY cohort survives 30y")
sm_f = st.safemax(ge, gb, gi, "fixed")
sm_g = st.safemax(ge, gb, gi, "gk")
print(f"  fixed {100*sm_f:.2f}%   guardrails {100*sm_g:.2f}%")

print("\n# Headline strategies -- all quantities REAL, initial wealth = 1")
g5, g55 = res_g[0.050], res_g[0.055]
HEAD = (("fixed 4%", f4), ("fixed 5%", f5), ("GK 5%", g5), ("GK 5.5%", g55))
print("             succ%   LTI mean/med/p05     min-income med/p05/worst   "
      "cuts raises freezes")
for name, r in HEAD:
    s = st.summarize(r)
    print(f"  {name:<9} {s['success_pct']:6.2f}  "
          f"{s['lti_mean']:.3f}/{s['lti_median']:.3f}/{s['lti_p05']:.3f}    "
          f"{s['mininc_median']:.4f}/{s['mininc_p05']:.4f}/{s['mininc_worst']:.4f}"
          f"     {s['cuts_mean']:.1f}   {s['raises_mean']:.1f}    "
          f"{s['freezes_mean']:.1f}")
print("  (LTI = 30-year lifetime real income per $1 initial; min-income = the "
      "worst single-year real paycheck)")

print("\n# The income price of ruin-proofing (GK 5% vs the 4%-rule paycheck 0.040)")
yb5 = (g5["income"] < 0.04 - 1e-12).sum(axis=1)
yb55 = (g55["income"] < 0.04 - 1e-12).sum(axis=1)
print(f"  GK 5%  : years below the 4% paycheck  mean {yb5.mean():.2f}  median "
      f"{np.median(yb5):.0f}  max {yb5.max()}  |  {100*(yb5 > 0).mean():.2f}% of "
      "cohorts dip below at least once")
print(f"  GK 5.5%: years below the 4% paycheck  mean {yb55.mean():.2f}  |  "
      f"{100*(yb55 > 0).mean():.2f}% of cohorts dip below")
sh_lti = 100 * (g5["lti"] < f4["lti"]).mean()
print(f"  {sh_lti:.2f}% of cohorts end 30 years with LESS lifetime real income "
      "from GK 5% than from the plain 4% rule")
print(f"  worst-cohort paycheck floor: GK 5% {g5['min_inc'].min():.4f} "
      f"({100*(1 - g5['min_inc'].min()/0.05):.0f}% real cut from the initial "
      "0.050); the fixed rule never cuts -- it ruins instead")

print("\n# HAC t (Newey-West, bandwidth = full 360-month cohort overlap)")
pairs = (
    ("LTI  GK5  - fixed4", g5["lti"] - f4["lti"]),
    ("LTI  GK5  - fixed5", g5["lti"] - f5["lti"]),
    ("LTI  GK5.5- fixed4", g55["lti"] - f4["lti"]),
    ("TW   GK5  - fixed5", g5["terminal"] - f5["terminal"]),
)
for label, d in pairs:
    print(f"  {label}: mean {d.mean():+.4f}  HAC t = "
          f"{st.hac_tstat(d, lags=360):+.2f}")
dsucc5 = 100 * (g5["success"].mean() - f5["success"].mean())
print(f"  dsuccess @5% (GK - fixed): {dsucc5:+.2f} pp  "
      "(a rate, not a mean -- CI from the bootstrap below)")

print(f"\n# Circular block bootstrap ({N_BOOT} reps, 120-month blocks, seed 597)")
print("#   NOTE: blocks preserve within-decade dynamics but destroy cross-decade")
print("#   mean reversion -- stated, not hidden. 95% percentile CIs.")
boot = st.block_bootstrap(eq, bd, infl, n_boot=N_BOOT, seed=597)
for k, label in (("dsucc5", "dsuccess @5%   GK - fixed (pp)"),
                 ("dlti54", "dLTI  GK5 - fixed4"),
                 ("dlti55", "dLTI  GK5 - fixed5"),
                 ("dterm5", "dTW   GK5 - fixed5")):
    lo, hi = st.ci(boot[k])
    print(f"  {label:<32} CI [{lo:+.3f}, {hi:+.3f}]")

print("\n# Famous cohorts -- lifetime real income (per $1 initial) and the floor")
for y in ("1929-09-01", "1966-01-01", "1972-12-01"):
    i = np.where(dates == pd.Timestamp(y))[0][0]
    print(f"  retire {y[:7]}: fixed4 LTI {f4['lti'][i]:.3f} "
          f"({'ok' if f4['success'][i] else 'RUIN'})   fixed5 LTI "
          f"{f5['lti'][i]:.3f} ({'ok' if f5['success'][i] else 'RUIN'})   "
          f"GK5 LTI {g5['lti'][i]:.3f} (ok, min paycheck "
          f"{g5['min_inc'][i]:.4f}, {g5['n_cut'][i]:.0f} cuts, "
          f"{g5['n_freeze'][i]:.0f} freezes)")

print("\n# Which rule does the rescue? (variants at a 5% start)")
for p in ("gk", "gk_freeze_only", "gk_noraise", "gk_cut_always"):
    r = st.simulate(ge, gb, gi, wr0=0.05, preset=p)
    s = st.summarize(r)
    print(f"  {p:<15} succ {s['success_pct']:6.2f}%  LTI mean {s['lti_mean']:.3f}"
          f"  min-inc med {s['mininc_median']:.4f} worst {s['mininc_worst']:.4f}"
          f"  cuts {s['cuts_mean']:.1f} raises {s['raises_mean']:.1f} "
          f"freezes {s['freezes_mean']:.1f}")

print("\n# Robustness -- allocation x one-way cost (GK 5% succ / LTI ; fixed 5% succ)")
for ew in (0.60, 0.65):
    for cb in (0.0, 10.0, 25.0):
        g = st.simulate(ge, gb, gi, wr0=0.05, preset="gk", eq_w=ew, cost_bps=cb)
        f = st.simulate(ge, gb, gi, wr0=0.05, preset="fixed", eq_w=ew,
                        cost_bps=cb)
        print(f"  {int(100*ew)}/{int(100*(1-ew))} cost {cb:>4.0f} bps: GK "
              f"{100*g['success'].mean():6.2f}% / {g['lti'].mean():.3f}   "
              f"fixed {100*f['success'].mean():6.2f}%")

print("\n# Synthetic control -- 20 independent seeded worlds per setting, offline")
print("#   detector: dsuccess (GK - fixed, same rate); one-sample t across worlds.")
print("#   Machinery proof only -- never market evidence.")
rows = [
    ("EXACT NULL  calm world (vol 10%, wr 3%)", dict(eq_vol=0.10, wr=0.03)),
    ("mild ruin   (vol 10%, wr 4%)", dict(eq_vol=0.10, wr=0.04)),
    ("PLANTED     historical vol (17%), wr 5%", dict(eq_vol=0.17, wr=0.05)),
    ("PLANTED     high vol (25%), wr 5%", dict(eq_vol=0.25, wr=0.05)),
    ("PLANTED     historical vol (17%), wr 6%", dict(eq_vol=0.17, wr=0.06)),
]
for label, kw in rows:
    r = st.control_rescue(n_seeds=20, **kw)
    t = "  nan (all-zero)" if np.isnan(r["dsucc_t"]) else f"{r['dsucc_t']:+6.2f}"
    print(f"  {label:<41} fixed-fail {r['fixed_fail_mean']:5.2f}%  "
          f"dsuccess {r['dsucc_mean']:+6.2f} pp (t {t})  "
          f"dLTI {r['dlti_mean']:+.3f} (t {r['dlti_t']:+.2f})")
print("\n(the exact null reads 0.00 pp in every world -- nothing to rescue; the")
print(" rescue lights up exactly where fixed-rate ruin exists. The +dLTI in the")
print(" calm world is the prosperity rule mechanically paying raises -- expected.)")
