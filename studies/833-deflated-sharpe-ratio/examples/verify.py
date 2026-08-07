"""Reproducible headline run for Study 833 — the Deflated Sharpe Ratio.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Fully deterministic, offline, no network — the "tape" is a
seeded null simulation.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from deflated_sharpe import data, strategy as st  # noqa: E402

N_TRIALS = 1000
N_DAYS = 1260
ANN_VOL = 0.15
SEED = 833

print("# Deflated Sharpe Ratio — does the trial count inflate the best Sharpe? (Bailey & LdP 2014)")
print(f"[sim] NULL world: {N_TRIALS} independent, true-zero-edge strategies x {N_DAYS} days "
      f"(ann vol {ANN_VOL:.0%}), seed {SEED}")
print(f"      as-of {data.AS_OF}   Fingerprint {data.config_fingerprint(N_TRIALS, N_DAYS, ANN_VOL, SEED)}")

# --------------------------------------------------------------------------- #
# 1. The headline — best of N empties vs the expected-maximum-Sharpe formula
# --------------------------------------------------------------------------- #
panel = data.null_panel(N_TRIALS, N_DAYS, ANN_VOL, SEED)
be = st.best_sharpe_experiment(panel)
print("\n# THE HEADLINE — best of 1,000 EMPTY strategies (true Sharpe = 0)")
print(f"  mean column Sharpe (the truth) : {be['mean_sharpe_ann']:+.3f}  (~0, nothing real)")
print(f"  observed MAX Sharpe (annualised): {be['obs_max_sharpe_ann']:+.3f}  <- 'a gorgeous backtest'")
print(f"  expected MAX under null  SR0    : {be['exp_max_sharpe_ann']:+.3f}  (the bar luck alone clears)")
print(f"  cross-trial SR std sqrt(V)      : {be['sr_std_across_trials']:.5f}  (theory 1/sqrt(T-1) = {1/np.sqrt(N_DAYS-1):.5f})")

# --------------------------------------------------------------------------- #
# 2. The deflation — DSR of that winner
# --------------------------------------------------------------------------- #
best = panel[:, be["best_col"]]
d = st.deflated_sharpe_ratio(best, N_TRIALS)
print("\n# THE DEFLATION — Deflated Sharpe Ratio of the winner")
print(f"  winner annualised Sharpe : {d['sharpe_ann']:+.3f}")
print(f"  expected-max bar  SR0     : {d['sr0_ann']:+.3f} (annualised)")
print(f"  deflated EXCESS Sharpe    : {d['deflated_excess_ann']:+.3f}  (~0: shrunk back to nothing)")
print(f"  DSR (prob. it beats luck) : {d['dsr']:.3f}  (< 0.95 => consistent with luck)")
print(f"  naive one-sample t        : {st.one_sample_t(best):+.2f}  (looks 'significant' — it is not)")

# --------------------------------------------------------------------------- #
# 3. Out-of-sample collapse (the Mirage picture) + costed timer
# --------------------------------------------------------------------------- #
champ = st.in_sample_champion(panel, frac=0.5)
print("\n# OUT-OF-SAMPLE COLLAPSE — pick IS champion, watch it live")
print(f"  in-sample Sharpe  : {champ['is_sharpe_ann']:+.3f}")
print(f"  out-of-sample Sharpe: {champ['oos_sharpe_ann']:+.3f}  (NW t = {champ['oos_t_nw']:+.2f})")
oos = panel[champ["is_n"]:, champ["champion"]]
for cb in (1.0, 5.0):
    tm = st.timer_stats(oos, cost_bps=cb)
    print(f"  timer OOS @ {cb:>3.0f} bp one-way: gross {tm['gross_bps']:+.3f} -> net "
          f"{tm['net_bps']:+.3f} bps/day (t = {tm['t_net']:+.2f})")

# --------------------------------------------------------------------------- #
# 4. The inflation curve — E[max] grows with N (observed vs formula)
# --------------------------------------------------------------------------- #
print("\n# THE INFLATION CURVE — best Sharpe climbs with N (40 seeds/point)")
ic = st.inflation_curve(n_days=N_DAYS, ann_vol=ANN_VOL, n_seeds=40, base_seed=SEED)
for N, o, sd, p in zip(ic["n_grid"], ic["obs_best_ann"], ic["obs_best_sd_ann"], ic["pred_ann"]):
    print(f"  N={int(N):>5}: observed best {o:+.3f} (sd {sd:.3f})   formula E[max] {p:+.3f}")

# --------------------------------------------------------------------------- #
# 5. Synthetic controls — the machinery is calibrated
# --------------------------------------------------------------------------- #
print("\n# NULL CALIBRATION — naive screen vs DSR on 40 empty pools (N=1000)")
cal = st.null_dsr_calibration(n_trials=N_TRIALS, n_days=N_DAYS, ann_vol=ANN_VOL,
                              n_seeds=40, base_seed=SEED)
print(f"  mean best annualised Sharpe : {cal['mean_best_sharpe_ann']:+.3f}")
print(f"  naive |t|>=2 fires          : {cal['naive_fire']}/{cal['n_seeds']} "
      f"({cal['naive_fire_rate']:.0%})  <- manufactures 'discoveries'")
print(f"  DSR>=0.95 fires             : {cal['dsr_fire']}/{cal['n_seeds']} "
      f"({cal['dsr_fire_rate']:.0%})  <- calibrated near nominal 5%")
print(f"  mean DSR                    : {cal['mean_dsr']:.3f}  (~0.5, coin flip)")
print(f"  mean deflated-excess Sharpe : {cal['mean_deflated_excess_ann']:+.3f}  (~0)")

print("\n# POSITIVE CONTROL — an HONEST single strategy keeps a high DSR (40 seeds)")
for tsr in (1.0,):
    hc = st.honest_control(true_ann_sharpe=tsr, n_days=N_DAYS, ann_vol=ANN_VOL,
                           n_seeds=40, base_seed=SEED)
    print(f"  true Sharpe {tsr:.1f}: realised {hc['mean_sharpe_ann']:+.3f}, "
          f"mean DSR {hc['mean_dsr']:.3f}, DSR>=0.95 in {hc['dsr_fire']}/{hc['n_seeds']} "
          f"({hc['dsr_fire_rate']:.0%})")

# A strong real strategy planted inside the N=1000 pool still survives.
pool, planted = data.planted_in_pool(N_TRIALS, N_DAYS, true_ann_sharpe=2.0, seed=SEED)
srs = st.panel_sr_per_period(pool)
jwin = int(np.nanargmax(srs))
d_plant = st.deflated_sharpe_ratio(pool[:, planted], N_TRIALS)
print(f"  planted true-Sharpe-2.0 col buried in N=1000: is the pool winner? "
      f"{'yes' if jwin == planted else 'no'}; its DSR = {d_plant['dsr']:.3f} "
      f"(survives the deflation even at N=1000)")
