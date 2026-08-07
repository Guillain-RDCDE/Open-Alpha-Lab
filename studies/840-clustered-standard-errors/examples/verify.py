"""Reproduce every number quoted in docs/results.md and the notebook ``R`` dict — Study 840
(Clustered Standard Errors / cross-sectional dependence). Pure simulation: no network, no
cache, deterministic under CONFIG['seed'].

    python studies/840-clustered-standard-errors/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clustered_se import data as d, strategy as st  # noqa: E402

cfg = d.CONFIG


def main() -> None:
    print("=" * 78)
    print("Study 840 — Clustered Standard Errors   (as-of", d.AS_OF + ")")
    print("Config fingerprint:", d.config_fingerprint())
    print("Config:", {k: cfg[k] for k in ("seed", "n_reps", "n_firms", "n_periods",
                                          "rho_x", "rho_e", "crit", "control_beta")})
    print("=" * 78)

    # --- HERO: the N=50, T=50, rho_x=rho_e=0.5 null panel -------------------
    X, Y = d.panel(cfg["n_reps"], cfg["n_periods"], cfg["n_firms"],
                   rho_x=cfg["rho_x"], rho_e=cfg["rho_e"], beta=0.0, seed=cfg["seed"])
    print("null-panel fingerprint:", d.fingerprint(X))
    c = st.calibration(X, Y, crit=cfg["crit"])
    moult = d.theoretical_moulton(cfg["n_firms"], cfg["rho_x"], cfg["rho_e"])
    print(f"\n--- HERO: N={cfg['n_firms']} firms, T={cfg['n_periods']} periods, "
          f"rho_x={cfg['rho_x']}, rho_e={cfg['rho_e']}, beta=0 (the null) ---")
    print(f"  nominal false-positive rate       : {c['nominal']:.4f}")
    print(f"  pooled-slope true SD (MC truth)   : {c['true_sd']:.5f}   "
          f"(point estimate mean {c['b_mean']:+.5f} -> unbiased)")
    print(f"  Fama-MacBeth slope true SD        : {c['true_sd_fm']:.5f}   "
          f"(FM point estimate mean {c['b_fm_mean']:+.5f})")
    print(f"  Moulton closed-form inflation     : sqrt(1+(N-1)*rho_x*rho_e) = {moult:.3f}")
    print("\n  estimator      FP rate   Wilson95            mean SE   SE/true   t SD")
    labels = [("NAIVE OLS", "ols"), ("firm-clustered", "firm"),
              ("time-clustered", "time"), ("Fama-MacBeth", "fm")]
    for name, k in labels:
        print(f"  {name:14s} {c[k+'_fp']:.4f}   "
              f"[{c[k+'_fp_ci'][0]:.3f}, {c[k+'_fp_ci'][1]:.3f}]     "
              f"{c[k+'_se_mean']:.5f}   {c[k+'_se_ratio']:.3f}    {c[k+'_t_sd']:.3f}")

    # --- CONTROL: switch the common time factor OFF (rho_x=rho_e=0) ---------
    ctl = st.iid_control(cfg["n_reps"], cfg["n_periods"], cfg["n_firms"],
                         seed=cfg["seed"], crit=cfg["crit"])
    print("\n--- CONTROL: rho_x = rho_e = 0 (no common time factor -> i.i.d. panel) ---")
    print(f"  NAIVE OLS  FP : {ctl['ols_fp']:.4f}   firm : {ctl['firm_fp']:.4f}   "
          f"time : {ctl['time_fp']:.4f}   FM : {ctl['fm_fp']:.4f}   (all ~ nominal)")
    print(f"  naive t SD    : {ctl['naive_t_sd']:.3f}   (~1: no inflation, SE ratio "
          f"{ctl['ols_se_ratio']:.3f})")

    # --- inflation curve vs rho_e ------------------------------------------
    df = st.inflation_curve_rho(cfg["rho_e_grid"], cfg["n_reps"], cfg["n_periods"],
                                cfg["n_firms"], cfg["rho_x"], seed=cfg["seed"], crit=cfg["crit"])
    print("\n--- INFLATION vs residual intra-period correlation rho_e (rho_x=0.5, N=50) ---")
    print("  rho_e  ols_fp firm_fp time_fp  fm_fp  naive_t_sd  Moulton")
    for _, r in df.iterrows():
        print(f"  {r['rho_e']:.2f}   {r['ols_fp']:.3f}  {r['firm_fp']:.3f}   "
              f"{r['time_fp']:.3f}  {r['fm_fp']:.3f}   {r['naive_t_sd']:8.3f}  {r['theory_moulton']:7.3f}")

    # --- inflation curve vs N ----------------------------------------------
    df2 = st.inflation_curve_nfirms(cfg["n_firms_grid"], cfg["n_reps"], cfg["n_periods"],
                                    cfg["rho_x"], cfg["rho_e"], seed=cfg["seed"], crit=cfg["crit"])
    print("\n--- INFLATION vs cross-section size N (rho_x=rho_e=0.5) ---")
    print("  N     ols_fp firm_fp time_fp  fm_fp  naive_t_sd  Moulton")
    for _, r in df2.iterrows():
        print(f"  {int(r['n_firms']):4d}  {r['ols_fp']:.3f}  {r['firm_fp']:.3f}   "
              f"{r['time_fp']:.3f}  {r['fm_fp']:.3f}   {r['naive_t_sd']:8.3f}  {r['theory_moulton']:7.3f}")

    # --- positive control: Fama-MacBeth still FIRES on a planted effect -----
    pw = st.power_check(cfg["n_reps"], cfg["n_periods"], cfg["n_firms"],
                        cfg["rho_x"], cfg["rho_e"], beta=cfg["control_beta"],
                        seed=cfg["seed"], crit=cfg["crit"])
    print(f"\n--- POSITIVE CONTROL: planted slope beta = {cfg['control_beta']} ---")
    print(f"  Fama-MacBeth rejection (power)    : {pw['fm_power']:.3f}")
    print(f"  mean Fama-MacBeth t               : {pw['fm_t_mean']:+.3f}")
    print(f"  share FM t > 0 (correct sign)     : {pw['fm_t_positive_share']:.3f}")
    print(f"  mean FM slope (recovers beta)     : {pw['b_fm_mean']:+.4f}")

    # --- the costed timer on one null panel --------------------------------
    Xo, Yo = d.one_panel(cfg["n_periods"], cfg["n_firms"], rho_x=cfg["rho_x"],
                         rho_e=cfg["rho_e"], beta=0.0, seed=cfg["seed"])
    tm = st.timer_stats(Xo, Yo, ret_scale=cfg["ret_scale"])
    print("\n--- TIMER on one null panel (nothing to trade, by construction) ---")
    print(f"  gross bps/period                  : {tm['gross_bps']:+.3f}")
    print(f"  net   bps/period (after cost+borrow): {tm['net_bps']:+.3f}")
    print(f"  net annualised %                  : {tm['ann_net_pct']:+.2f}")


if __name__ == "__main__":
    main()
