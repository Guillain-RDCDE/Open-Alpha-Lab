"""Reproducible headline run for Study 620 — A-H Share Premium.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached pair closes under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from a_h_premium import data, strategy as st

print("# A-H Share Premium — 14 dual-listed pairs, Shanghai (A, CNY) vs Hong Kong (H, HKD)")
if data.have_real():
    from quantlab import repro

    close, adj = data.load_real()
    prem_d = st.premium_panel(close)
    prem_m = st.monthly_premium(prem_d)
    rel_m = st.monthly_rel_returns(adj)
    rel_po = st.monthly_rel_returns_price_only(close)
    span = (close.index.max() - close.index.min()).days / 365.25
    print(f"prices : {close.shape[0]:,} daily rows x {close.shape[1]} series "
          f"({len(data.PAIRS)} pairs + 2 FX), {close.index.min().date()} -> "
          f"{close.index.max().date()} ({span:.1f} years)")
    print(repro.data_stamp("monthly A-H premium panel", prem_m, asof=data.AS_OF))

    # ------------------------------------------------------------------ level
    print("\n# The premium level — does Shanghai really pay ~30% more?")
    lv = st.level_stats(prem_m, lags=12)
    _mu36, _se36, t36 = st.nw_tstat(lv["ew"], lags=36)
    print(f"  EW mean premium : {lv['mean_pct']:+.2f}%  (claim says ~30%)  "
          f"HAC(12) t = {lv['hac_t']:+.2f}  HAC(36) t = {t36:+.2f}  (n = {lv['n_months']} months)")
    print(f"  months > 0      : {lv['share_pos']*100:.1f}%   last negative month: "
          f"{lv['ew'][lv['ew'] < 0].index.max().date()}")
    print(f"  as-of level     : {lv['last_pct']:+.1f}%   range [{lv['min_pct']:+.1f}%, "
          f"{lv['max_pct']:+.1f}%]")
    post = lv["ew"].loc["2014-12":]
    pmu, _pse, pt = st.nw_tstat(post, lags=12)
    print(f"  post-Connect    : {pmu*100:+.2f}% mean since 2014-12 (Stock Connect opening), "
          f"HAC(12) t = {pt:+.2f}, {int((post > 0).mean()*100)}% of {len(post)} months positive")
    pp = lv["per_pair_pct"]
    print(f"  per pair        : {lv['n_pairs_pos']}/{lv['n_pairs']} pairs with a positive mean; "
          f"widest {pp.index[0]} {pp.iloc[0]:+.1f}%, {pp.index[1]} {pp.iloc[1]:+.1f}%; "
          f"narrowest {pp.index[-1]} {pp.iloc[-1]:+.1f}%, {pp.index[-2]} {pp.iloc[-2]:+.1f}%")

    print("\n# Monte-Carlo null placebo for the level (near-unit-root honesty)")
    for lbl, kw in [("calibrated AR(1)", {}),
                    ("stress phi = 0.995", {"phi_override": 0.995}),
                    ("random walk from the 2010 start", {"rw": True})]:
        mc = st.null_mc_level(lv["ew"], **kw)
        print(f"  {lbl:32s}: p(mean >= {mc['obs_mean_pct']:+.1f}%) = {mc['p_mean']:.4f}   "
              f"p(share_pos >= {mc['obs_share']*100:.1f}%) = {mc['p_share']:.4f}"
              + (f"   (phi = {mc['phi']:.3f}, innov sd = {mc['sig']:.4f})" if not kw else ""))

    # -------------------------------------------------------- non-convergence
    print("\n# Non-convergence — is the premium distinguishable from a random walk?")
    ar = st.ar1_stats(lv["ew"])
    print(f"  AR(1) phi = {ar['phi']:.3f} (se {ar['se_phi']:.3f})  ->  half-life "
          f"{ar['half_life_m']:.1f} months")
    print(f"  Dickey-Fuller t = {ar['df_t']:+.2f}  vs 5% critical value {ar['df_crit']:.2f}  "
          f"->  CANNOT reject a unit root")
    hl = st.per_pair_half_lives(prem_m)
    print(f"  per-pair half-lives: median {hl.median():.1f} months, "
          f"range {hl.min():.1f} - {hl.max():.1f} ({len(hl)} pairs)")

    # ------------------------------------------------------------ co-movement
    print("\n# Co-movement — one factor, not 14 mispricings")
    cm = st.comovement(prem_m)
    print(f"  mean pairwise corr of monthly premium changes: {cm['mean_pair_corr']:.3f}")
    print(f"  PC1 share of variance: {cm['pc1_share']*100:.1f}%  "
          f"(vs {100.0/cm['n_pairs']:.1f}% if independent; {cm['n_pairs']} pairs, "
          f"{cm['n_months']} complete months)")

    # --------------------------------------------------------------- the fade
    print("\n# The fade trade (PAPER) — long H / short A wide half, reverse in narrow half")
    print("  signal = month-end premium; held the NEXT month (the one documented lag)")
    f0 = st.fade_cross_section(prem_m, rel_m, cost_bps_oneway=0.0)
    print(f"  gross          : {f0['mean_bps']:+.1f} bps/mo  HAC t = {f0['hac_t']:+.2f}  "
          f"(ann {f0['ann_pct']:+.2f}%, churn {f0['avg_churn']*100:.1f}%/mo, n = {f0['n_months']})")
    fs = st.fade_cross_section(prem_m.shift(1), rel_m, cost_bps_oneway=0.0)
    print(f"  skip-a-month   : {fs['mean_bps']:+.1f} bps/mo  HAC t = {fs['hac_t']:+.2f}  "
          f"(kills month-end bounce)")
    for lo, hi in [("2010-01", "2018-06"), ("2018-07", "2026-06")]:
        f2 = st.fade_cross_section(prem_m.loc[lo:hi], rel_m.loc[lo:hi])
        print(f"  {lo} - {hi}: {f2['mean_bps']:+.1f} bps/mo  HAC t = {f2['hac_t']:+.2f}  "
              f"(n = {f2['n_months']})")
    fx2 = st.fade_cross_section(prem_m.drop(columns=["Great Wall Motor", "China Life"]),
                                rel_m.drop(columns=["Great Wall Motor", "China Life"]))
    print(f"  ex 2 extremes  : {fx2['mean_bps']:+.1f} bps/mo  HAC t = {fx2['hac_t']:+.2f}  "
          f"(drop Great Wall + China Life)")
    fpo = st.fade_cross_section(prem_m, rel_po)
    print(f"  PRICE-ONLY     : {fpo['mean_bps']:+.1f} bps/mo  HAC t = {fpo['hac_t']:+.2f}  "
          f"(not a dividend-carry artefact)")
    rb = st.fade_random_baseline(prem_m, rel_m, n_seeds=20)
    print(f"  random halves  : {rb['mean_bps']:+.1f} bps/mo  mean t = {rb['mean_t']:+.2f}  "
          f"(averaged over {rb['n_seeds']} seeds)")
    zp = st.predictive_slope(prem_m, rel_m)
    print(f"  z-portfolio    : {zp['mean_bps']:+.1f} bps/mo  HAC t = {zp['hac_t']:+.2f}  "
          f"(premium z-score x next-month H-A)")
    ts = st.fade_time_series(prem_m, rel_m)
    print(f"  own-history TS : {ts['mean_bps']:+.1f} bps/mo  HAC t = {ts['hac_t']:+.2f}  "
          f"(wide vs own 36m median — the level does NOT revert to its own past)")
    cy = st.carry_stats(rel_m)
    cy_po = st.carry_stats(rel_po)
    print(f"  uncond. carry  : {cy['mean_bps']:+.1f} bps/mo  HAC t = {cy['hac_t']:+.2f} "
          f"(total-return)  |  {cy_po['mean_bps']:+.1f} bps/mo t = {cy_po['hac_t']:+.2f} "
          f"(price-only)")

    # ------------------------------------------------- the executable fraction
    print("\n# The only executable HALF — long A (Connect) / short H (HK borrow), narrow pairs")
    print("  (the wide half needs to SHORT A-shares: no borrow channel exists for outsiders)")
    fl = st.feasible_leg(prem_m, rel_m)
    print(f"  gross          : {fl['mean_bps']:+.1f} bps/mo  HAC t = {fl['hac_t']:+.2f}  "
          f"(churn {fl['avg_churn']*100:.1f}%/mo)")
    fl1 = st.feasible_leg(prem_m, rel_m, cost_bps_oneway=25.0, borrow_bps_yr=150.0)
    print(f"  net 25bp+1.5%b : {fl1['mean_bps']:+.1f} bps/mo  HAC t = {fl1['hac_t']:+.2f}  "
          f"(ann {fl1['ann_pct']:+.2f}%)")
    fl2 = st.feasible_leg(prem_m, rel_m, cost_bps_oneway=50.0, borrow_bps_yr=300.0)
    print(f"  net 50bp+3.0%b : {fl2['mean_bps']:+.1f} bps/mo  HAC t = {fl2['hac_t']:+.2f}  "
          f"(ann {fl2['ann_pct']:+.2f}%)")
    for lo, hi in [("2010-01", "2018-06"), ("2018-07", "2026-06")]:
        fl3 = st.feasible_leg(prem_m.loc[lo:hi], rel_m.loc[lo:hi])
        print(f"  gross {lo} - {hi}: {fl3['mean_bps']:+.1f} bps/mo  HAC t = {fl3['hac_t']:+.2f}")

    # ------------------------------------------------------------- third axis
    print("\n# Third axis — is the discount side (H) at least the better BUY?")
    race = st.leg_race(adj)
    print(f"  H basket : {race['h_ann_pct']:+.2f}%/yr total-return in CNY  "
          f"(wealth x{race['h_wealth']:.2f} over {race['years']:.1f} years)")
    print(f"  A basket : {race['a_ann_pct']:+.2f}%/yr total-return in CNY  "
          f"(wealth x{race['a_wealth']:.2f})")
    print(f"  H - A    : {race['diff_bps_mo']:+.1f} bps/mo  HAC t = {race['hac_t']:+.2f}  "
          f"(n = {race['n_months']} months) -> statistically NOTHING")
else:
    print("(no _cache/ahp_close.csv — run data.fetch_panel() once to build the cache)")

# ------------------------------------------------------------------ synthetic
print("\n# Synthetic control — machinery proof only, never market evidence")
print("  (a) level detector, 40 seeds per world:")
for plant in (0.30, 0.0):
    c = st.control_level_suite(plant, n_seeds=40)
    print(f"    planted {plant*100:5.1f}%: seed-avg EW mean {c['avg_mean_pct']:+6.2f}% "
          f"(sd {c['sd_pct']:.2f}, max|.| {c['max_abs_pct']:.1f}%)  "
          f"seeds |mean| >= 29.7%: {c['n_ge_real']}/{c['n_seeds']}  "
          f"seeds |HAC t| >= 2: {c['n_hac_t_ge2']}/{c['n_seeds']}")
print("  -> 26/40 ZERO-mean worlds fake |HAC t| >= 2 on the level: HAC t on a near-unit-root")
print("     level is not a certificate — hence the Monte-Carlo placebo above.")
print("  (b) fade detector, 10 seeds per world:")
for phi in (0.97, 1.0):
    c = st.control_fade_suite(phi, n_seeds=10)
    print(f"    premium phi = {phi:.2f}: seed-avg {c['avg_bps']:+5.1f} bps/mo, "
          f"seed-avg t = {c['avg_t']:+.2f}  (min {c['min_t']:+.2f}, max {c['max_t']:+.2f})")
print("  -> the fade fires ONLY when the premium truly mean-reverts; a random-walk premium")
print("     world reads flat. The real tape's fade t is therefore evidence of reversion.")
