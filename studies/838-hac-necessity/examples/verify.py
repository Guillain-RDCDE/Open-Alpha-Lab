"""Reproduce every number quoted in docs/results.md and the notebook ``R`` dict — Study 838
(HAC Necessity). Pure simulation: no network, no cache, deterministic under CONFIG['seed'].

    python studies/838-hac-necessity/examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hac_necessity import data as d, strategy as st  # noqa: E402

cfg = d.CONFIG


def main() -> None:
    print("=" * 74)
    print("Study 838 — HAC Necessity   (as-of", d.AS_OF + ")")
    print("Config fingerprint:", d.config_fingerprint())
    print("Config:", {k: cfg[k] for k in ("seed", "n_reps", "n_days", "window",
                                          "nw_lags", "crit", "control_mean")})
    print("=" * 74)

    # --- HERO: the 21-day overlapping null ---------------------------------
    X = d.overlap_matrix(cfg["n_reps"], cfg["n_days"], window=cfg["window"],
                         seed=cfg["seed"], daily_vol=cfg["daily_vol"], mean=0.0)
    print("null-matrix fingerprint:", d.fingerprint(X))
    fp = st.false_positive_rate(X, lags=cfg["nw_lags"], crit=cfg["crit"])
    print("\n--- HERO: overlapping window = 21 days, mean = 0 (the null) ---")
    print(f"  nominal false-positive rate      : {fp['nominal']:.4f}")
    print(f"  NAIVE  false-positive rate       : {fp['naive_fp']:.4f}  "
          f"Wilson95 [{fp['naive_fp_ci'][0]:.3f}, {fp['naive_fp_ci'][1]:.3f}]")
    print(f"  NEWEY-WEST false-positive rate   : {fp['nw_fp']:.4f}  "
          f"Wilson95 [{fp['nw_fp_ci'][0]:.3f}, {fp['nw_fp_ci'][1]:.3f}]")
    print(f"  naive t SD (~inflation factor)   : {fp['naive_t_sd']:.3f}   "
          f"(closed form sqrt(21) = {d.theoretical_inflation_overlap(21):.3f})")
    print(f"  NW t SD (~1 if calibrated)       : {fp['nw_t_sd']:.3f}")

    # --- the no-autocorrelation control (window=1) -------------------------
    ctl = st.iid_control(cfg["n_reps"], cfg["n_days"], seed=cfg["seed"],
                         daily_vol=cfg["daily_vol"], lags=cfg["nw_lags"], crit=cfg["crit"])
    print("\n--- CONTROL: window = 1 (i.i.d., no serial correlation) ---")
    print(f"  NAIVE  false-positive rate       : {ctl['naive_fp']:.4f}   (back at nominal)")
    print(f"  NEWEY-WEST false-positive rate   : {ctl['nw_fp']:.4f}")
    print(f"  naive t SD                       : {ctl['naive_t_sd']:.3f}   (~1: no inflation)")

    # --- inflation curve vs the overlap window -----------------------------
    ov = st.inflation_curve_overlap(cfg["window_grid"], cfg["n_reps"], cfg["n_days"],
                                    seed=cfg["seed"], daily_vol=cfg["daily_vol"], crit=cfg["crit"])
    print("\n--- INFLATION CURVE vs overlap window (NW lags = 2*window) ---")
    print("  window  naive_fp   nw_fp  naive_t_sd  theory=sqrt(w)")
    for _, r in ov.iterrows():
        print(f"  {int(r['window']):5d}   {r['naive_fp']:.3f}    {r['nw_fp']:.3f}   "
              f"{r['naive_t_sd']:8.3f}   {r['theory_inflation']:8.3f}")

    # --- inflation curve vs AR(1) rho --------------------------------------
    ar = st.inflation_curve_ar1(cfg["rho_grid"], cfg["n_reps"], cfg["n_days"],
                                seed=cfg["seed"], daily_vol=cfg["daily_vol"],
                                lags=st.nw_auto_lags(cfg["n_days"]), crit=cfg["crit"])
    print(f"\n--- INFLATION CURVE vs AR(1) rho (NW auto lags = {st.nw_auto_lags(cfg['n_days'])}) ---")
    print("  rho    naive_fp   nw_fp  naive_t_sd  theory=sqrt((1+r)/(1-r))")
    for _, r in ar.iterrows():
        print(f"  {r['rho']:.2f}   {r['naive_fp']:.3f}    {r['nw_fp']:.3f}   "
              f"{r['naive_t_sd']:8.3f}   {r['theory_inflation']:8.3f}")

    # --- positive control: the machinery FIRES on a planted effect ---------
    pw = st.power_check(cfg["n_reps"], cfg["n_days"], window=cfg["window"],
                        mean=cfg["control_mean"], seed=cfg["seed"],
                        daily_vol=cfg["daily_vol"], lags=cfg["nw_lags"], crit=cfg["crit"])
    print(f"\n--- POSITIVE CONTROL: planted daily mean = {cfg['control_mean']:.1e} ---")
    print(f"  Newey-West rejection (power)     : {pw['nw_power']:.3f}")
    print(f"  mean NW t                        : {pw['nw_t_mean']:.2f}")
    print(f"  share NW t > 0 (correct sign)    : {pw['nw_t_positive_share']:.3f}")

    # --- the costed timer on one null path ---------------------------------
    tm = st.timer_stats(d.overlap_returns(cfg["n_days"], cfg["window"], seed=cfg["seed"]).values)
    print("\n--- TIMER on one null path (nothing to trade, by construction) ---")
    print(f"  gross bps/day                    : {tm['gross_bps']:+.3f}")
    print(f"  net   bps/day (after cost+borrow): {tm['net_bps']:+.3f}")
    print(f"  net annualised %                 : {tm['ann_net_pct']:+.2f}")


if __name__ == "__main__":
    main()
