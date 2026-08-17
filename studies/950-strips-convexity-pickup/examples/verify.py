"""Real-tape verification — Study 950 (Zero-Coupon Convexity). Regenerates docs/results.md.

Reads cached daily total-return closes for the zero-coupon Treasury funds (EDV, ZROZ), the
coupon long bond (TLT) and the cash leg (BIL), plus the 30-year constant-maturity yield
(^TYX). Duration-matches the coupon mix to each zero fund on the realised beta to the same
rate factor, races them excess-of-cash, and runs the asymmetry (convexity) regression, the
move-size buckets, the block-bootstrap CIs, the era cut and the cost / financing sweeps.
Network only on ``--fetch``.

    python studies/950-strips-convexity-pickup/examples/verify.py            # cache-only
    python studies/950-strips-convexity-pickup/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from zero_convexity import data, strategy as st  # noqa: E402

ZERO = "EDV"
CROSS = "ZROZ"
COUPON = "TLT"
CASH = "BIL"
YIELD = "TYX"
WINDOW = 252
COST_BPS = 3.0
FINANCE_BPS = 25.0   # PROXY/ASSUMPTION — swept below


def legs(px, zero_ticker):
    z = px[zero_ticker].dropna()
    c = px[COUPON].dropna()
    k = px[CASH].dropna()
    y = px[YIELD].dropna()
    return z, c, k, y


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    z, c, k, y = legs(px, ZERO)
    race = st.run_race(z, c, k, y, window=WINDOW, cost_bps=COST_BPS, finance_bps=FINANCE_BPS)
    m = st.to_monthly(race, y)

    used = px.loc[race.index, [ZERO, COUPON, CASH, YIELD]]
    print(f"{ZERO} vs duration-matched {COUPON}+{CASH}:  "
          f"{race.index[0].date()} -> {race.index[-1].date()}  n={len(race):,} days / "
          f"{len(m)} months  fp={data.fingerprint(used)}")
    print(f"as-of {data.AS_OF}   window={WINDOW}d   cost={COST_BPS} bps one-way   "
          f"financing spread={FINANCE_BPS} bps/yr (PROXY)")
    print(f"hedge ratio L on {COUPON}: mean {race['L'].mean():.3f}  "
          f"sd {race['L'].std():.3f}  range [{race['L'].min():.2f}, {race['L'].max():.2f}]")

    s_zero, s_mix, s_diff = (st.summary(race[c_]) for c_ in ("e_zero", "e_mix", "diff"))
    print("\n=== the duration-matched race (excess-of-cash, daily) ===")
    for tag, s in [(f"A: 100% {ZERO}", s_zero), (f"B: {COUPON}+{CASH} mix", s_mix),
                   ("A - B (the spread)", s_diff)]:
        print(f"  {tag:22s} mean {s['mean_ann']:+.2%}/yr  vol {s['vol_ann']:.2%}  "
              f"Sharpe {s['sharpe']:+.3f}  HAC t {s['tstat']:+.2f}")
    print(f"  vol ratio A/B = {s_zero['vol_ann']/s_mix['vol_ann']:.3f} "
          f"(1.00 = a clean duration match)")
    print(f"  monthly spread: mean {m['diff'].mean()*1e4:+.2f} bp/mo  "
          f"HAC t {st.newey_west_t(m['diff'].to_numpy(), lags=6):+.2f}  "
          f"hit rate {(m['diff'] > 0).mean():.1%}")

    print("\n=== the headline: asymmetry (convexity) regression, monthly ===")
    for reg_kind, label in [("dy2", "squared NET monthly move  dy^2"),
                            ("rv", "realised variance  sum(dy_t^2)")]:
        r = st.convexity_regression(m, regressor=reg_kind)
        print(f"  [{label}]  n={r['n']}  R2={r['r2']:.3f}")
        print(f"    a  = {r['a_bp_mo']:+7.2f} bp/mo   (t {r['a_t']:+.2f})   <- the price of convexity")
        print(f"    b1 = {r['b1']:+7.4f}            (t {r['b1_t']:+.2f})   "
              f"residual duration {r['b1_resid_duration_yrs']:+.2f} yr")
        print(f"    b2 = {r['b2']:+7.1f}            (t {r['b2_t']:+.2f})   "
              f"= {r['b2_per_25bp']:+.2f} bp at a 25 bp move, "
              f"{r['b2_per_50bp']:+.2f} bp at 50 bp")
        print(f"    breakeven monthly move = {r['breakeven_move_bp']:.1f} bp "
              f"(median |dy| = {m['dy'].abs().median()*1e4:.1f} bp)")

    print("\n=== move-size buckets (terciles of |monthly yield move|) ===")
    tbl = st.move_buckets(m)
    for lab, row in tbl.iterrows():
        print(f"  {lab:9s} n={int(row['n']):3d}  mean |dy| {row['mean_absdy_bp']:5.1f} bp  "
              f"spread {row['mean_diff_bp_mo']:+7.2f} bp/mo  (t {row['t_hac']:+.2f})  "
              f"hit {row['hit_rate']:.1%}")
    print("  after removing the residual linear rate exposure (in-sample, descriptive):")
    for lab, row in st.move_buckets(m, col="diff_hedged").iterrows():
        print(f"  {lab:9s} n={int(row['n']):3d}                      "
              f"spread {row['mean_diff_bp_mo']:+7.2f} bp/mo  (t {row['t_hac']:+.2f})  "
              f"hit {row['hit_rate']:.1%}")

    print("\n=== block bootstrap (2,000 draws, 6-month blocks) ===")
    ci_mean = st.block_bootstrap_ci(m["diff"], stat="mean", seed=950)
    ci_shp = st.block_bootstrap_ci(m["diff"], stat="sharpe", seed=950)
    for kind, r in [("mean spread", ci_mean)]:
        print(f"  {kind:12s}: {r['point']*1e4:+.2f} bp/mo  95% CI "
              f"[{r['ci_low']*1e4:+.2f}, {r['ci_high']*1e4:+.2f}]  share<0 {r['frac_negative']:.1%}")
    print(f"  spread Sharpe: {ci_shp['point']:+.3f}  95% CI "
          f"[{ci_shp['ci_low']:+.3f}, {ci_shp['ci_high']:+.3f}]  share<0 {ci_shp['frac_negative']:.1%}")
    for kind in ("dy2", "rv"):
        b = st.bootstrap_b2_ci(m, seed=950, regressor=kind)
        print(f"  b2 [{kind:3s}]  : {b['point']:+.1f}  95% CI [{b['ci_low']:+.1f}, "
              f"{b['ci_high']:+.1f}]  share<0 {b['frac_negative']:.1%}")

    print("\n=== era cut (split 2018-01-01) ===")
    eras = st.era_cut(race, y, split="2018-01-01")
    for tag, e in eras.items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_months']:3d} mo  spread {e['mean_diff_bp_mo']:+7.2f} bp/mo "
              f"(t {e['t_diff']:+.2f})  a {e['a_bp_mo']:+7.2f} (t {e['a_t']:+.2f})  "
              f"b2 {e['b2']:+7.1f} (t {e['b2_t']:+.2f})  b1 t {e['b1_t']:+.2f}")

    print("\n=== cost sweep (one-way bps on the mix's rebalance turnover) ===")
    for row in st.sweep_costs(z, c, k, y, finance_bps=FINANCE_BPS, window=WINDOW):
        print(f"  cost={row['cost_bps']:5.1f} bps  spread {row['mean_diff_bp_mo']:+7.2f} bp/mo "
              f"(t {row['t_diff']:+.2f})  b2 {row['b2']:+7.1f} (t {row['b2_t']:+.2f})")

    print("\n=== financing-spread sweep (the PROXY; a wider spread FLATTERS the zero) ===")
    for row in st.sweep_finance(z, c, k, y, cost_bps=COST_BPS, window=WINDOW):
        print(f"  spread={row['finance_bps']:6.1f} bps/yr  spread {row['mean_diff_bp_mo']:+7.2f} bp/mo "
              f"(t {row['t_diff']:+.2f})  b2 {row['b2']:+7.1f} (t {row['b2_t']:+.2f})")

    print("\n=== hedge-window sweep (how the duration match is solved) ===")
    for row in st.sweep_window(z, c, k, y, cost_bps=COST_BPS, finance_bps=FINANCE_BPS):
        print(f"  window={row['window']:4d}d  n={row['n_months']:3d} mo  L {row['L_mean']:.3f}  "
              f"spread {row['mean_diff_bp_mo']:+7.2f} bp/mo (t {row['t_diff']:+.2f})  "
              f"b2 {row['b2']:+7.1f} (t {row['b2_t']:+.2f})")

    print(f"\n=== cross-check: {CROSS} (PIMCO 25+ year zeros) ===")
    z2, c2, k2, y2 = legs(px, CROSS)
    race2 = st.run_race(z2, c2, k2, y2, window=WINDOW, cost_bps=COST_BPS, finance_bps=FINANCE_BPS)
    m2 = st.to_monthly(race2, y2)
    r2 = st.convexity_regression(m2, regressor="dy2")
    r2rv = st.convexity_regression(m2, regressor="rv")
    print(f"  {race2.index[0].date()} -> {race2.index[-1].date()}  {len(m2)} months  "
          f"L mean {race2['L'].mean():.3f}  vol ratio "
          f"{race2['e_zero'].std()/race2['e_mix'].std():.3f}")
    print(f"  spread {m2['diff'].mean()*1e4:+.2f} bp/mo "
          f"(t {st.newey_west_t(m2['diff'].to_numpy(), lags=6):+.2f})")
    print(f"  b2[dy2] {r2['b2']:+.1f} (t {r2['b2_t']:+.2f})   "
          f"b2[rv] {r2rv['b2']:+.1f} (t {r2rv['b2_t']:+.2f})   "
          f"a {r2rv['a_bp_mo']:+.2f} bp/mo (t {r2rv['a_t']:+.2f})")
    tbl2 = st.move_buckets(m2)
    for lab, row in tbl2.iterrows():
        print(f"    {lab:9s} spread {row['mean_diff_bp_mo']:+7.2f} bp/mo (t {row['t_hac']:+.2f})")

    print("\n=== FULL cut grid: 2 funds x 3 eras x 2 quadratic specs (nothing omitted) ===")
    grid = st.cut_grid({ZERO: race, CROSS: race2}, y, split="2018-01-01")
    print(f"  {'fund':5s} {'era':6s} {'spec':4s} {'n':>4s} {'a bp/mo':>9s} {'t':>7s} "
          f"{'b2':>9s} {'t':>7s} {'b1 t':>7s}")
    for _, r in grid.iterrows():
        flag = "  <-- |t|>=2" if abs(r["b2_t"]) >= 2 else ("  <-- SIGN FLIP" if r["b2"] <= 0 else "")
        print(f"  {r['fund']:5s} {r['era']:6s} {r['spec']:4s} {int(r['n_months']):4d} "
              f"{r['a_bp_mo']:+9.2f} {r['a_t']:+7.2f} {r['b2']:+9.1f} {r['b2_t']:+7.2f} "
              f"{r['b1_t']:+7.2f}{flag}")
    cen = st.grid_census(grid)
    print(f"  census: b2 > 0 in {cen['b2_positive']}/{cen['n_cuts']} cuts, "
          f"a < 0 in {cen['a_negative']}/{cen['n_cuts']}; "
          f"|t| >= 2 in {cen['n_b2_t_ge_2']}/{cen['n_cuts']}; "
          f"largest |t| anywhere {cen['max_b2_t']:+.2f} ({cen['max_cut']})")

    print("\n=== synthetic control (machinery proof only - never supports the stamp) ===")
    pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=950)[0])
    print(f"  planted pickup : b2[rv] {pl['b2']:+.1f} (t {pl['b2_t']:+.2f})  "
          f"a {pl['a_bp_mo']:+.2f} bp/mo (t {pl['a_t']:+.2f})  vol ratio {pl['vol_ratio']:.3f}")
    nulls = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=950 + s)[0])["b2_t"]
        for s in range(8)
    ])
    print(f"  null x8        : b2[rv] t mean {nulls.mean():+.2f} (sd {nulls.std(ddof=1):.2f}), "
          f"|t|>=2 in {(np.abs(nulls) >= 2).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
