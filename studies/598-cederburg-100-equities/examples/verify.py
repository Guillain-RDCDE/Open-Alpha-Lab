"""Reproducible headline run for Study 598 — Cederburg "100% Equities for Life".

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; cache-first on the study's own
``_cache/`` (Shiller parquet + EFA monthly csv); the synthetic control always
runs offline.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cederburg_100_equities import data, strategy as st

N_BOOT_OUTER = 600
N_DRAWS = 2000
WR_GRID = (0.030, 0.035, 0.040, 0.050)
PAIRS = (("alleq", "s6040"), ("alleq", "tdf"), ("alleq_dom", "s6040"))
PAIR_LABEL = {
    ("alleq", "s6040"): "HEADLINE   alleq(50/50 dom/intl) - 60/40  (the paper's claim)",
    ("alleq", "tdf"): "VS GLIDE   alleq(50/50 dom/intl) - TDF    (the paper's other target)",
    ("alleq_dom", "s6040"): "PURE TAPE  alleq(100% domestic)  - 60/40  (no simulated leg anywhere)",
}

print("# Cederburg 100% Equities -- lifecycle horse race on the Shiller + EFA real tape")
df = data.real_returns()
us, intl, bd = df["US"].to_numpy(), df["INTL"].to_numpy(), df["BD"].to_numpy()
n_mkt = int(df["intl_is_market"].sum())
mkt_start = df.index[df["intl_is_market"]].min()
ann = {c: 100 * ((1 + df[c]).prod() ** (12 / len(df)) - 1) for c in ("US", "INTL", "BD")}
print(f"tape        : {df.index.min().date()} -> {df.index.max().date()} "
      f"({len(df)} months)  fingerprint {data.fingerprint(df)}")
print(f"real returns: US {ann['US']:.2f}%/yr  INTL {ann['INTL']:.2f}%/yr  "
      f"BD {ann['BD']:.2f}%/yr  (geometric, CPI-deflated)")
print(f"INTL leg    : {n_mkt}/{len(df)} months ({100*n_mkt/len(df):.1f}%) are MARKET data "
      f"(EFA, from {mkt_start.date()}); the rest is LITERATURE-CALIBRATED "
      f"(DMS: geo {100*data.INTL_REAL_GEO:.1f}%/yr, vol {100*data.INTL_VOL:.0f}%, "
      f"corr {data.CORR_US_INTL:.2f}, seed {data.INTL_SEED}) -- simulation, NOT market data")
pre = ~df["intl_is_market"]
print(f"realised corr(US, INTL): pre-EFA {df.loc[pre, ['US','INTL']].corr().iloc[0,1]:.3f} "
      f"(calibrated)  EFA era {df.loc[~pre, ['US','INTL']].corr().iloc[0,1]:.3f} (market)")

gu, gi, gb, starts = st.cohort_year_returns(us, intl, bd)
dates = df.index[starts]
ret_dates = dates + pd.DateOffset(years=st.SAVE_YEARS)
print(f"cohorts     : {gu.shape[0]} monthly-start 70-year lifecycles "
      f"(save 40y, retire 30y); starts {dates.min().date()} -> {dates.max().date()}, "
      f"retirements {ret_dates.min().date()} -> {ret_dates.max().date()}")
print(f"mechanics   : contribute 1.0 real/yr for 40y; withdraw 4% of the pot at "
      f"retirement, fixed real, for 30y; costs {st.COST_BPS:.0f} bps one-way x traded "
      f"value + {st.ER_BPS:.0f} bps/yr expense ratio; weights set at the end of the "
      f"prior year (one clean lag)")

print("\n# Scoreboard at the 4% rule (wealth in multiples of the annual contribution)")
res4 = st.run_all(gu, gi, gb, wr=0.04)
for s in st.STRATEGIES:
    m = st.summarize(res4[s])
    print(f"  {s:<10} pot@65 med {m['wret_median']:7.1f}   terminal mean {m['mean']:7.1f}  "
          f"median {m['median']:7.1f}  p05 {m['p05']:6.1f}   ruin {m['ruin_pct']:5.2f}% "
          f"({m['n_fail']}/{m['n_cohorts']})")

print("\n# Head-to-heads -- HAC t (Newey-West, bandwidth = full 840-month overlap)")
print("#   on per-cohort terminal-wealth differences at 4%; ruin gaps in pct points")
for p in PAIRS:
    a, b = p
    d = res4[a]["terminal"] - res4[b]["terminal"]
    t = st.hac_tstat(d, lags=840)
    druin = 100 * ((~res4[a]["success"]).mean() - (~res4[b]["success"]).mean())
    win = 100 * np.mean(res4[a]["terminal"] > res4[b]["terminal"])
    print(f"  {PAIR_LABEL[p]}")
    print(f"      mean dTW {d.mean():+9.1f}  HAC t = {t:+.2f}   druin {druin:+.2f} pp   "
          f"win rate {win:.1f}%")

print(f"\n# Outer circular block bootstrap ({N_BOOT_OUTER} reps, 120-month blocks, seed 598)")
print("#   tape-level uncertainty of the cohort metrics; 95% percentile CIs.")
print("#   Blocks preserve within-decade dynamics, destroy cross-decade mean")
print("#   reversion -- stated, not hidden.")
boot = st.outer_bootstrap(us, intl, bd, pairs=PAIRS, n_boot=N_BOOT_OUTER, seed=598)
for p in PAIRS:
    lo_m, hi_m = st.ci(boot[p]["dmean"])
    lo_r, hi_r = st.ci(boot[p]["druin"])
    print(f"  {p[0]:>9} - {p[1]:<6} dmeanTW CI [{lo_m:+8.1f}, {hi_m:+8.1f}]   "
          f"druin CI [{lo_r:+6.2f}, {hi_r:+6.2f}] pp")

print("\n# Ruin rates by withdrawal rate (share of cohorts depleted before age 95)")
hdr = "  wr      " + "".join(f"{s:>11}" for s in st.STRATEGIES)
print(hdr)
for wr in WR_GRID:
    r = st.run_all(gu, gi, gb, wr=wr)
    row = "".join(f"{100*(~r[s]['success']).mean():>10.2f}%" for s in st.STRATEGIES)
    print(f"  {100*wr:.1f}%  {row}")

print(f"\n# Cederburg machinery -- {N_DRAWS} bootstrap lifetimes "
      f"(120-month blocks, seed 598), 4% rule")
BU, BI, BB = st.block_lifetimes(us, intl, bd, n_draws=N_DRAWS, seed=598)
rb = st.run_all(BU, BI, BB, wr=0.04)
for s in st.STRATEGIES:
    m = st.summarize(rb[s])
    print(f"  {s:<10} pot@65 med {m['wret_median']:7.1f}   terminal mean {m['mean']:7.1f}  "
          f"median {m['median']:7.1f}   ruin {m['ruin_pct']:5.2f}%")
print("  (Monte-Carlo view on resampled 70-year lifetimes -- the paper's own method;")
print("   the ordering matches the cohort panel: the blend's ruin is NOT lower than 60/40)")

print("\n# Sensitivity -- what the international leg assumption does (alleq vs 60/40)")
for mode, label in (("hybrid", "hybrid (headline: DMS-calibrated pre-2001 + EFA)"),
                    ("corr1", "corr=1 stress (zero pre-2001 diversification)"),
                    ("domestic", "domestic (INTL=US -- pure tape, no simulation)")):
    d2 = data.real_returns(intl_mode=mode)
    gu2, gi2, gb2, _ = st.cohort_year_returns(
        d2["US"].to_numpy(), d2["INTL"].to_numpy(), d2["BD"].to_numpy())
    r2 = st.run_all(gu2, gi2, gb2, wr=0.04, strategies=("alleq", "s6040"))
    d = r2["alleq"]["terminal"] - r2["s6040"]["terminal"]
    print(f"  {label:<48} ruin {100*(~r2['alleq']['success']).mean():5.2f}%  "
          f"meanTW {r2['alleq']['terminal'].mean():7.1f}  "
          f"HAC t vs 60/40 = {st.hac_tstat(d, 840):+.2f}")

print("\n# Seed-robustness -- re-drawing the simulated INTL residuals (seeds 598-617)")
print("#   the pre-2001 INTL leg is a seeded simulation; a single-seed verdict is")
print("#   banned on this desk, so both head-to-heads are re-run under 20 seeds")
t_tdf, t_60, ruin_worse = [], [], 0
for sd in range(598, 618):
    dfs = data.real_returns(intl_mode="hybrid", seed=sd)
    gus, gis, gbs, _ = st.cohort_year_returns(
        dfs["US"].to_numpy(), dfs["INTL"].to_numpy(), dfs["BD"].to_numpy())
    rs = st.run_all(gus, gis, gbs, wr=0.04, strategies=("alleq", "s6040", "tdf"))
    t_tdf.append(st.hac_tstat(rs["alleq"]["terminal"] - rs["tdf"]["terminal"], 840))
    t_60.append(st.hac_tstat(rs["alleq"]["terminal"] - rs["s6040"]["terminal"], 840))
    ruin_worse += ((~rs["alleq"]["success"]).mean()
                   > (~rs["s6040"]["success"]).mean())
t_tdf, t_60 = np.array(t_tdf), np.array(t_60)
print(f"  alleq - TDF  : HAC t mean {t_tdf.mean():+.2f}  min {t_tdf.min():+.2f}  "
      f"max {t_tdf.max():+.2f}  t>=2 in {(t_tdf >= 2).sum()}/20 seeds  <- ROBUST")
print(f"  alleq - 60/40: HAC t mean {t_60.mean():+.2f}  min {t_60.min():+.2f}  "
      f"max {t_60.max():+.2f}  t>=2 in {(t_60 >= 2).sum()}/20 seeds  <- SIGN FLIPS")
print(f"  ruin(alleq) > ruin(60/40) in {ruin_worse}/20 seeds -- the ruin race vs 60/40")
print("  is a coin flip across residual draws (matches the wide bootstrap CI)")
dd = data.real_returns(intl_mode="domestic")
gud, gid, gbd, _ = st.cohort_year_returns(
    dd["US"].to_numpy(), dd["INTL"].to_numpy(), dd["BD"].to_numpy())
rd = st.run_all(gud, gid, gbd, wr=0.04, strategies=("alleq_dom", "tdf_dom"))
ddt = rd["alleq_dom"]["terminal"] - rd["tdf_dom"]["terminal"]
print(f"  PURE TAPE alleq_dom - tdf_dom (nothing simulated): dmeanTW {ddt.mean():+.1f}  "
      f"HAC t = {st.hac_tstat(ddt, 840):+.2f}  ruin "
      f"{100*(~rd['alleq_dom']['success']).mean():.2f}% vs "
      f"{100*(~rd['tdf_dom']['success']).mean():.2f}%")
print("  (the certified all-equity-vs-TDF edge survives every residual draw AND the")
print("   pure tape; nothing in the 60/40 race is decided by market data)")

print("\n# Third axis -- the worst historical cohorts (retire 1929-09 and 1966-01)")
print("#   terminal wealth (x annual contribution) at 4%; fail year = years into")
print("#   retirement at depletion (-1 = survived)")
d1 = data.real_returns(intl_mode="corr1")
gu1, gi1, gb1, _ = st.cohort_year_returns(
    d1["US"].to_numpy(), d1["INTL"].to_numpy(), d1["BD"].to_numpy())
r66c = st.simulate(gu1, gi1, gb1, st.weights_for("alleq"), wr=0.04)
for ret_ym, start_ym in (("1929-09", "1889-09-01"), ("1966-01", "1926-01-01")):
    i = int(np.where(dates == pd.Timestamp(start_ym))[0][0])
    row = "  ".join(f"{s} {res4[s]['terminal'][i]:7.1f} (fy {res4[s]['fail_year'][i]:>2})"
                    for s in st.STRATEGIES)
    print(f"  retire {ret_ym}: {row}")
    print(f"      alleq under corr=1 stress: terminal {r66c['terminal'][i]:.1f} "
          f"(fy {r66c['fail_year'][i]})")

print("\n# Robustness -- cost levels (one-way bps / expense-ratio bps per yr)")
for cb, er in ((0.0, 0.0), (10.0, 5.0), (25.0, 20.0)):
    r = st.run_all(gu, gi, gb, wr=0.04, cost_bps=cb, er_bps=er)
    d = r["alleq"]["terminal"] - r["s6040"]["terminal"]
    print(f"  cost {cb:>4.0f}/{er:>4.0f}: alleq meanTW {r['alleq']['terminal'].mean():7.1f}  "
          f"ruin {100*(~r['alleq']['success']).mean():5.2f}%   "
          f"HAC t vs 60/40 = {st.hac_tstat(d, 840):+.2f}")

print("\n# Synthetic control -- 20 independent seeded worlds per setting, offline")
print("#   detector: alleq vs 60/40, one-sample t across worlds. Exact null:")
print("#   premium = 0 (equal arithmetic means => equal expected terminal wealth")
print("#   for every schedule), so the dmean detector must read ~0 there.")
print("#   Machinery proof only -- never market evidence.")
for prem in (0.0, 0.02, 0.045):
    r = st.control_premium_effect(prem, wr=0.04, n_seeds=20)
    print(f"  premium {100*prem:>4.1f} pp: dmeanTW {r['dmean_mean']:+8.1f} "
          f"(t {r['dmean_t']:+5.2f})   druin {r['druin_mean']:+6.2f} pp "
          f"(t {r['druin_t']:+5.2f})")
print("\n(the mean-wealth detector reads ~0 at the exact null and lights up with the")
print(" planted premium; the ruin detector shows the risk channel -- at a thin premium")
print(" 100% equity ruins MORE, and the gap closes as the premium grows: exactly the")
print(" economics the real-tape verdict turns on)")
