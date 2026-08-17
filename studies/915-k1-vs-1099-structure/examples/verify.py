"""Real-tape verification — Study 915 (K-1 vs 1099). Regenerates docs/results.md numbers.

Reads cached daily total-return closes for the two wrappers (DBC = Schedule K-1 commodity
pool, PDBC = Form 1099 fund), the cash leg (BIL) and the three context vehicles, then runs
the pre-tax race (excess-of-cash Sharpe, HAC *t* on the daily difference, paired
block-bootstrap CIs, tracking error, era cut, cost sweep, calendar-year sign test) and the
after-tax bracket x payout sweep. Network only on ``--fetch``.

    python studies/915-k1-vs-1099-structure/examples/verify.py            # cache-only
    python studies/915-k1-vs-1099-structure/examples/verify.py --fetch    # refresh tapes
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wrapper_tax import data, strategy as st  # noqa: E402

K1 = data.WRAPPER_K1        # DBC  — the Schedule K-1 commodity pool
RIC = data.WRAPPER_1099     # PDBC — the Form 1099 "No K-1" twin
CASH = data.CASH            # BIL  — cash leg and collateral-interest proxy
CONTEXT = ("DBC", "PDBC", "USCI", "USO", "BNO")


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    pair = px[[RIC, K1, CASH]].dropna()
    print(f"{RIC} (1099) vs {K1} (K-1), cash {CASH}: "
          f"{pair.index[0].date()} -> {pair.index[-1].date()}  n={len(pair)}  "
          f"fp={data.fingerprint(pair)}")
    print(f"as-of {data.AS_OF}  |  expense-ratio PROXY: "
          f"{K1} {data.EXPENSE_RATIO_PROXY[K1]:.2%}, {RIC} {data.EXPENSE_RATIO_PROXY[RIC]:.2%} "
          f"(fee gap favours {RIC} by "
          f"{(data.EXPENSE_RATIO_PROXY[K1] - data.EXPENSE_RATIO_PROXY[RIC])*1e4:.0f} bp/yr)")

    res = st.race(px, RIC, K1, CASH)

    def fmt(s, cagr, dd):
        return (f"exSharpe={s['sharpe']:+.3f}  CAGR={cagr:+.2%}  vol={s['vol_ann']:.1%}  "
                f"MaxDD={dd:+.1%}  HAC t={s['tstat']:+.2f}")

    print("\n=== pre-tax race (pure tape, both legs excess-of-cash) ===")
    print(f"  {RIC:5s} (1099): {fmt(res['a'], res['cagr_a'], res['dd_a'])}")
    print(f"  {K1:5s} (K-1) : {fmt(res['b'], res['cagr_b'], res['dd_b'])}")
    trk = res["tracking"]
    print(f"\n  difference (1099 - K-1): {res['diff_ann']:+.4%}/yr "
          f"({trk['diff_bps_day']:+.4f} bp/day)  HAC t = {res['t_diff']:+.2f}")
    print(f"  tracking error {res['te_ann']:.2%}/yr  corr {trk['corr']:.4f}  "
          f"beta {trk['beta']:.4f}  R2 {trk['r2']:.4f}")
    print(f"  excess-of-cash Sharpe advantage: {res['sharpe_adv']:+.4f}")

    print("\n=== paired block bootstrap (4,000 draws, 21-day blocks) ===")
    ci = st.bootstrap_diff_ci(res["r_a"], res["r_b"], seed=915)
    adv = st.bootstrap_sharpe_adv_ci(res["e_a"], res["e_b"], seed=915)
    print(f"  annualised return difference: {ci['point']:+.4%}  "
          f"95% CI [{ci['ci_low']:+.4%}, {ci['ci_high']:+.4%}]  P(<0)={ci['frac_negative']:.3f}")
    print(f"  Sharpe advantage            : {adv['point']:+.4f}  "
          f"95% CI [{adv['ci_low']:+.4f}, {adv['ci_high']:+.4f}]  P(<0)={adv['frac_negative']:.3f}")

    print("\n=== is the daily tracking error economic, or microstructure? ===")
    mon = st.horizon_tracking(px, RIC, K1, freq="ME")
    print(f"  daily-scaled TE {mon['te_ann_daily_scaled']:.2%}/yr  vs  monthly TE "
          f"{mon['te_ann']:.2%}/yr over {mon['n_obs']} months")
    print(f"  lag-1 autocorrelation of the daily difference: {mon['daily_autocorr_lag1']:+.3f} "
          f"(the two closes disagree, then re-converge)")
    print(f"  monthly difference {mon['diff_ann']:+.4%}/yr (HAC t = {mon['t_diff']:+.2f})")

    print("\n=== power — how large a wrapper cost could this sample have caught? ===")
    pw = st.power_stats(res["r_a"], res["r_b"], boot_sd_ann=ci["boot_sd"])
    print(f"  annualised SE of the difference: i.i.d. {pw['se_iid_ann']:.2%}, "
          f"HAC {pw['se_hac_ann']:.2%}, bootstrap {pw['se_boot_ann']:.2%}")
    print(f"  minimum detectable difference at |t|=2: i.i.d. {pw['mde_iid_ann']:.2%}/yr, "
          f"HAC {pw['mde_hac_ann']:.2%}/yr, bootstrap {pw['mde_boot_ann']:.2%}/yr")

    print("\n=== era cut (split 2021-01-01) ===")
    for tag, e in st.era_cut(px, RIC, K1, CASH, split="2021-01-01").items():
        if e is None:
            continue
        print(f"  {tag:5s} {e['start']} -> {e['end']} (n={e['n_days']}): "
              f"diff {e['diff_ann']:+.3%}/yr (t={e['t_diff']:+.2f})  TE {e['te_ann']:.2%}  "
              f"Sharpe {e['sharpe_b']:+.3f} -> {e['sharpe_a']:+.3f}")

    print("\n=== cost sweep (extra one-way spread charged to the 1099 leg only) ===")
    for row in st.cost_sweep(px, RIC, K1, CASH):
        print(f"  +{row['extra_spread_bps']:5.1f} bp one-way  -> {row['annual_drag_bps']:5.2f} bp/yr drag  "
              f"diff {row['diff_ann']:+.4%}/yr (t={row['t_diff']:+.2f})  "
              f"Sharpe adv {row['sharpe_adv']:+.4f}")

    print("\n=== calendar-year table (complete years only) ===")
    tbl = st.calendar_year_table(px, RIC, K1, CASH)
    for y, row in tbl.iterrows():
        print(f"  {y}: {RIC} {row['a_ret']*100:+6.2f}%   {K1} {row['b_ret']*100:+6.2f}%   "
              f"diff {row['diff_pp']:+5.2f} pp   cash {row['cash_ret']*100:+4.2f}%")
    sgn = st.sign_test(tbl["diff_pp"])
    print(f"  1099 wins {sgn['wins']}/{sgn['n']} years (win rate {sgn['win_rate']:.0%}, "
          f"95% CI [{sgn['ci_low']:.0%}, {sgn['ci_high']:.0%}])")

    print("\n=== after-tax MODEL — every input below is an ASSUMPTION, not a measurement ===")
    print("  K-1 leg : Section 1256, 60/40 blend, annual mark-to-market, no deferral,")
    print("            basis stepped up each year -> nothing owed at liquidation.")
    print("  1099 leg: ordinary distributions = payout_share x BIL's total return, taxed")
    print("            annually; the rest deferred, taxed at LTCG on sale at the end.")
    grid = st.bracket_sweep(tbl)
    for tag in grid["bracket"].unique():
        sub = grid[grid["bracket"] == tag]
        cells = "  ".join(f"p={p:.1f}:{g:+.3f}" for p, g in
                          zip(sub["payout_share"], sub["gap_pp"]))
        print(f"  {tag:28s} blend {sub['blended_1256'].iloc[0]:.3f}  gap pp/yr -> {cells}")
    print(f"  gap range across the whole grid: [{grid['gap_pp'].min():+.3f}, "
          f"{grid['gap_pp'].max():+.3f}] pp/yr — sign flips: "
          f"{bool((grid['gap_pp'] > 0).any() and (grid['gap_pp'] < 0).any())}")

    print("\n=== the tax regime alone (both regimes applied to the SAME K-1 tape) ===")
    for tag, o, l in st.BRACKETS:
        r1 = st.same_tape_regime_gap(tbl, o, l, payout_share=1.0)
        print(f"  {tag:28s} K-1 {r1['cagr_k1']*100:+.3f}%  1099 {r1['cagr_ric']*100:+.3f}%  "
              f"regime gap {r1['regime_gap_pp']:+.3f} pp/yr")

    print("\n=== context: what ELSE the same investor is choosing between ===")
    disp = st.dispersion_table(px, CONTEXT, CASH)
    for leg, row in disp.iterrows():
        print(f"  {leg:5s}: CAGR {row['cagr']:+6.2%}  vol {row['vol_ann']:5.1%}  "
              f"exSharpe {row['excess_sharpe']:+.3f}  MaxDD {row['max_drawdown']:+.1%}")
    print(f"  index/vehicle dispersion: {(disp['cagr'].max()-disp['cagr'].min())*100:.2f} pp/yr "
          f"vs a wrapper difference of {abs(res['diff_ann'])*100:.2f} pp/yr")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    planted, truth = data.synthetic_daily(signal_strength=1.0, seed=915)
    d = st.synthetic_detect(planted)
    print(f"  planted drag {truth['planted_drag_ann']:.2%}/yr -> estimated "
          f"{d['estimated_drag_ann']:.2%}/yr  (HAC t = {d['t_diff']:+.2f}, TE {d['te_ann']:.2%})")
    import numpy as np
    nulls = np.array([
        st.race(p, "ric", "k1", "cash")["diff_ann"]
        for p, _ in data.synthetic_panel(n_pairs=8, signal_strength=0.0, seed=915)
    ])
    ts = np.array([
        st.race(p, "ric", "k1", "cash")["t_diff"]
        for p, _ in data.synthetic_panel(n_pairs=8, signal_strength=0.0, seed=915)
    ])
    print(f"  null x8: mean drift {nulls.mean():+.3%}/yr (sd {nulls.std(ddof=1):.3%}), "
          f"|t|>=2 in {(np.abs(ts) >= 2).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
