"""Reproducible headline run for Study 617 — Crash-Insurance-Cost.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached TAIL/IEF/SPY/^VIX tape
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crash_insurance_cost import data, strategy as st

print("# Crash-Insurance-Cost — TAIL / IEF / SPY / ^VIX (yfinance, total-return where applicable)")
if data.have_real():
    tape = data.load_real()
    try:
        from quantlab import repro
        fp = repro.fingerprint(tape)
    except Exception:
        fp = "n/a"
    print(f"tape        : {len(tape)} days, {tape.index[0].date()} -> {tape.index[-1].date()}"
          f"  as-of {data.AS_OF}  fingerprint {fp}")

    print("\n# 1 - Raw drift: what buy-and-hold crash insurance costs (TAIL, net of its 0.59%/yr ER)")
    d = st.drift_stats(tape)
    print(f"  window     : {d['start']} -> {d['end']}  ({d['years']:.1f} years, "
          f"{d['n_days']} daily returns, {d['n_months']} complete months)")
    print(f"  total ret  : {d['total_pct']:+.1f}%   CAGR {d['cagr_pct']:+.2f}%/yr   "
          f"ann vol {d['ann_vol_pct']:.1f}%")
    print(f"  daily mean : {d['daily_mean_bps']:+.2f} bps/day   HAC t = {d['daily_t']:+.2f} "
          f"(NW lags {d['nw_lags']})")
    print(f"  monthly    : {d['monthly_mean_bps']:+.2f} bps/mo    HAC t = {d['monthly_t']:+.2f} "
          f"(n = {d['n_months']})")

    print("\n# 2 - Decomposition vs its own collateral (alpha of TAIL on IEF = the put sleeve's drag)")
    dd = st.decompose_vs_ief(tape)
    dm = st.decompose_monthly(tape)
    print(f"  daily      : alpha {dd['alpha_daily_bps']:+.2f} bps/d = {dd['alpha_ann_pct']:+.2f}%/yr   "
          f"HAC t = {dd['t_alpha']:+.2f}   beta(IEF) {dd['beta_ief']:.2f} (t {dd['t_beta']:+.1f})   "
          f"R2 {dd['r2']:.3f}")
    print(f"  monthly    : alpha {dm['alpha_monthly_bps']:+.2f} bps/mo = {dm['alpha_ann_pct']:+.2f}%/yr   "
          f"HAC t = {dm['t_alpha']:+.2f}   beta(IEF) {dm['beta_ief']:.2f}   (n = {dm['n_months']} months)")
    sp = st.alpha_subperiods(tape)
    print(f"  pre-COVID  (-> 2020-02-19): alpha {sp['pre']['alpha_ann_pct']:+.2f}%/yr  "
          f"HAC t = {sp['pre']['t_alpha']:+.2f}  (n = {sp['pre']['n']})")
    print(f"  post-COVID (2020-07-01 ->): alpha {sp['post']['alpha_ann_pct']:+.2f}%/yr  "
          f"HAC t = {sp['post']['t_alpha']:+.2f}  (n = {sp['post']['n']})")

    print("\n# 3 - The 2020 payoff: the one time the insurance paid (and what happened next)")
    cv = st.covid_episode(tape)
    print(f"  TAIL gain from SPY peak (2020-02-19) to its own high ({cv['tail_peak_date']}): "
          f"{cv['tail_gain_pct']:+.2f}%")
    print(f"  SPY over the crash (2020-02-19 -> 2020-03-23): {cv['spy_fall_pct']:+.2f}%")
    print(f"  jackpot fully given back: first close below the pre-crash level on "
          f"{cv['giveback_date']}  ({cv['giveback_days']} days after the peak)")
    print(f"  at-inception (2017) holder: {cv['holder_at_peak_pct']:+.2f}% at the COVID peak  ->  "
          f"{cv['holder_now_pct']:+.2f}% at the end of tape")

    print("\n# 4 - The premium named: buyer-side variance premium from real ^VIX + SPY (model arithmetic)")
    vp = st.variance_premium(tape)
    print(f"  window     : {vp['start']} -> {vp['end']}  ({vp['n_months']} months; IV = prior "
          f"month-end VIX^2/12, one clean one-month lag)")
    print(f"  RV - IV    : {vp['mean_var_ann']*100:+.2f} var pts (ann)   HAC t = {vp['t_var']:+.2f}   "
          f"buyer wins {vp['share_buyer_wins']:.1f}% of months")
    print(f"  vol points : {vp['mean_vol_pts']:+.2f} pts (sqrt scale)    HAC t = {vp['t_vol']:+.2f}")
    print(f"  avg implied {vp['iv_ann_vol']:.2f}% vs realized {vp['rv_ann_vol']:.2f}% (ann vol)")
    vp2 = st.variance_premium(tape, start="2017-05-01")
    print(f"  TAIL era   : {vp2['start']} -> {vp2['end']}  RV-IV {vp2['mean_var_ann']*100:+.2f} var pts "
          f"(t = {vp2['t_var']:+.2f})   vol pts {vp2['mean_vol_pts']:+.2f} (t = {vp2['t_vol']:+.2f})   "
          f"buyer wins {vp2['share_buyer_wins']:.1f}%")

    print("\n# 5 - Third axis: has ANY buy-and-hold TAIL cohort ever come out ahead?")
    co = st.cohort_table(tape)
    print(f"  {co['n_cohorts']} month-end entry cohorts, each held to {d['end']}:")
    print(f"  ahead today: {co['n_ahead_now']} of {co['n_cohorts']} ({co['share_ahead_now']:.1f}%)  "
          f"best {co['best_pct']:+.2f}% (entered {co['best_date']})")
    print(f"  median {co['median_pct']:+.1f}%   worst {co['worst_pct']:+.1f}%")
    print(f"  EVER ahead at any month-end along the way (the COVID spike counts): "
          f"{co['share_ever_ahead']:.1f}% of cohorts")

    print("\n# 6 - Tradability: the insurance inside a portfolio (monthly rebalance, 5 bps one-way x turnover)")
    for w in (0.05, 0.10, 0.20):
        h = st.hedged_portfolio(tape, w_tail=w, cost_bps=5.0)
        print(f"  {int((1-w)*100)}/{int(w*100)} SPY/TAIL: CAGR {h['port_cagr_pct']:+.2f}%/yr vs "
              f"100% SPY {h['spy_cagr_pct']:+.2f}%/yr (drag {h['cagr_drag_pct']:.2f} pp/yr) | "
              f"maxDD {h['port_mdd_pct']:.1f}% vs {h['spy_mdd_pct']:.1f}% | "
              f"COVID DD {h['port_covid_dd_pct']:.1f}% vs {h['spy_covid_dd_pct']:.1f}%")
else:
    print("(no _cache/cic_tape.csv — run data.fetch_tape() once to build the cache)")

print("\n# 7 - Synthetic positive control (deterministic, no network, 20 seeds per setting)")
print("  the alpha-vs-bonds detector must stay quiet on FAIR insurance (bleed = 0) and must")
print("  recover a PLANTED bleed in size and light up in t.")
for row in st.control_summary(data.synthetic_world, bleeds=(0.0, 0.05), n_seeds=20):
    print(f"  planted bleed = {row['bleed_pct']:.0f}%/yr: mean alpha {row['mean_alpha_ann_pct']:+.2f}%/yr  "
          f"mean HAC t = {row['mean_t']:+.2f}  flagged {row['share_flagged']:.0f}% of seeds  "
          f"(n_seeds = {row['n_seeds']})")
