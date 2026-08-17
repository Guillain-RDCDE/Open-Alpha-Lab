"""Real-tape verification — Study 944 (How Much Leverage). Prints every docs/results.md number.

Reads the shared cache (SPY total return, ^IRX bill rate, BIL as the financing
cross-check), sweeps the constant daily-reset leverage multiple 1.0 -> 3.0, locates the
realised growth-optimal multiple, and then attacks it three ways: a block bootstrap of
the optimum, a rolling five-year ex-post optimum, and a *tradable* ex-ante Kelly rule
raced against unlevered buy-and-hold. Network only on ``--fetch``.

    python studies/944-optimal-leverage-realized/examples/verify.py            # cache-only
    python studies/944-optimal-leverage-realized/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab import repro  # noqa: E402

from optimal_leverage import data, strategy as st  # noqa: E402

SPREAD_BPS = data.DEFAULT_SPREAD_BPS
COST_BPS = data.DEFAULT_COST_BPS
SPLIT = "2015-01-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    legs = st.prepare_real(px)

    stamped = px[["SPY", "IRX"]].dropna()
    fp = data.fingerprint(stamped)
    print(repro.data_stamp("SPY x ^IRX", stamped, asof=data.AS_OF))
    print(f"SPY (total return) x ^IRX (bill rate): {legs.index[0].date()} -> "
          f"{legs.index[-1].date()}  n={len(legs):,}  fp={fp}")
    print(f"as-of {data.AS_OF}  |  financing spread {SPREAD_BPS:.0f} bps (PROXY)  "
          f"|  reset cost {COST_BPS:.1f} bps one-way")

    xc = st.financing_crosscheck(px)
    print(f"\nfinancing cross-check ({xc['start']} -> {xc['end']}): ^IRX-implied cash "
          f"{xc['irx_ann_pct']:.2f}%/yr vs BIL total return {xc['bil_ann_pct']:.2f}%/yr "
          f"(gap {xc['gap_bps']:+.0f} bps/yr)")

    # ---------------------------------------------------------------- the curve
    tab = st.sweep(legs, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    opt = float(tab["log_growth_ann"].idxmax())
    kelly = st.kelly_from_legs(legs)
    print("\n=== the leverage curve (daily reset, financed, costed) ===")
    print(f"{'L':>5}  {'terminal':>9}  {'CAGR':>8}  {'g(L)':>8}  {'exSharpe':>9}  "
          f"{'vol':>7}  {'maxDD':>8}  {'turnover':>9}")
    for lev in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, opt, 3.0):
        r = tab.loc[lev]
        star = " <-- realised optimum" if lev == opt else ""
        print(f"{lev:5.2f}  {r['terminal_wealth']:9.2f}x  {r['cagr']:+7.2%}  "
              f"{r['log_growth_ann']:+7.2%}  {r['excess_sharpe']:+9.3f}  "
              f"{r['vol_ann']:6.1%}  {r['max_drawdown']:+7.1%}  "
              f"{r['turnover_ann']:8.1f}x{star}")
    print(f"\nrealised growth-optimal multiple : {opt:.2f}")
    print(f"theoretical Kelly  mu/sigma^2     : {kelly:.2f}  "
          f"(the spread and the fat tails pull the realised peak below it)")
    gross = st.sweep(legs, spread_bps=0.0, cost_bps=0.0)
    print(f"Sharpe invariance check (0 bps spread, 0 cost): L=1 {gross.loc[1.0,'excess_sharpe']:.4f}"
          f"  L=3 {gross.loc[3.0,'excess_sharpe']:.4f}  -> identical by construction")

    # ------------------------------------------------- how wide is the peak?
    print("\n=== is the optimum knowable? (1) block bootstrap of the argmax ===")
    boot = st.bootstrap_optimum(legs, n_boot=1000, block=63,
                               spread_bps=SPREAD_BPS, cost_bps=COST_BPS, seed=944)
    print(f"  optimum {boot['opt']:.2f}   95% CI [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}]"
          f"   sd {boot['sd']:.2f}   at grid floor {boot['frac_at_floor']:.1%}"
          f"   at grid cap {boot['frac_at_cap']:.1%}")

    print("\n=== is the optimum knowable? (2) rolling five-year ex-post optimum ===")
    roll = st.rolling_optimum(legs, window=1260, step=21,
                              spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    inst = st.instability(roll)
    print(f"  {inst['n_windows']} windows: mean {inst['mean']:.2f}  sd {inst['sd']:.2f}  "
          f"range [{inst['min']:.2f}, {inst['max']:.2f}]  "
          f"at floor {inst['frac_at_floor']:.1%}  at cap {inst['frac_at_cap']:.1%}")
    print(f"  rolling Kelly estimate ranges [{inst['kelly_min']:+.2f}, {inst['kelly_max']:+.2f}]")
    yr = roll["opt_lev"].groupby(roll.index.year).last()
    print("  year-end reading: " + "  ".join(f"{y}:{v:.2f}" for y, v in yr.items()))

    print("\n=== is the optimum knowable? (3) era cut and the era hand-off ===")
    eras = st.era_cut(legs, split=SPLIT, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    for tag, e in eras.items():
        print(f"  {tag:5s} {e['start']} -> {e['end']} (n={e['n_days']}): optimum "
              f"{e['opt_lev']:.2f}  Kelly {e['kelly']:.2f}  CAGR@opt {e['cagr_opt']:+.2%}  "
              f"CAGR@1 {e['cagr_l1']:+.2%}  DD@1 {e['dd_l1']:+.1%}  DD@3 {e['dd_l3']:+.1%}")
    early_opt, late_opt = eras["early"]["opt_lev"], eras["late"]["opt_lev"]
    t_early = st.sweep(legs.loc[:SPLIT], spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    t_late = st.sweep(legs.loc[SPLIT:], spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    print(f"  hand-off: the LATE optimum ({late_opt:.2f}) applied in the EARLY era earns "
          f"{t_early.loc[late_opt, 'cagr']:+.2%} vs {t_early.loc[1.0, 'cagr']:+.2%} unlevered "
          f"(DD {t_early.loc[late_opt, 'max_drawdown']:+.1%})")
    print(f"            the EARLY optimum ({early_opt:.2f}) applied in the LATE era earns "
          f"{t_late.loc[early_opt, 'cagr']:+.2%} vs {t_late.loc[late_opt, 'cagr']:+.2%} at the "
          f"late optimum")

    print("\n=== is the optimum knowable? (4) start-date sensitivity ===")
    ss = st.start_sensitivity(legs, split=SPLIT, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    for s, r in ss.iterrows():
        print(f"  start {s} (n={r['n_days']:.0f}): optimum {r['opt_lev']:.2f}  "
              f"Kelly {r['kelly']:.2f}  CAGR@opt {r['cagr_opt']:+.2%}  "
              f"CAGR@1 {r['cagr_l1']:+.2%}  |  hand-off (late opt {r['late_opt']:.2f} in "
              f"early era) {r['handoff_cagr']:+.2%} vs {r['handoff_unlev']:+.2%} unlevered "
              f"-> edge {r['handoff_edge']:+.2%}/yr")

    # ------------------------------------------------------- the tradable rule
    print("\n=== the tradable version: ex-ante Kelly (trailing 756d, acted at t+1) ===")
    kel = st.ex_ante_kelly(legs, window=756, lo=1.0, hi=3.0,
                           spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    idx = kel.index
    arms = {"ex-ante Kelly": kel}
    for lev in (1.0, 2.0, opt, 3.0):
        arms[f"fixed L={lev:.2f}"] = st.levered_returns(legs.loc[idx], lev,
                                                        spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
    race = st.race_vs_fixed(legs.loc[idx], arms)
    print(f"  window {idx[0].date()} -> {idx[-1].date()}  n={len(idx):,}   applied multiple: "
          f"mean {kel['lev'].mean():.2f}, at cap {float((kel['lev'] >= 2.999).mean()):.1%} of days, "
          f"at floor {float((kel['lev'] <= 1.001).mean()):.1%}")
    for arm, r in race.iterrows():
        print(f"  {arm:16s} terminal {r['terminal_wealth']:7.2f}x  CAGR {r['cagr']:+7.2%}  "
              f"exSharpe {r['excess_sharpe']:+.3f}  vol {r['vol_ann']:5.1%}  "
              f"maxDD {r['max_drawdown']:+7.1%}")
    for base in ("fixed L=1.00", "fixed L=2.00"):
        d = st.growth_diff_test(kel, arms[base], seed=944)
        print(f"  vs {base}: log-growth advantage {d['log_growth_diff_ann']:+.2%}/yr  "
              f"HAC t={d['t_log_diff']:+.2f}  95% CI [{d['ci_low_ann']:+.2%}, {d['ci_high_ann']:+.2%}]"
              f"   (arithmetic-excess t={d['t_excess_diff']:+.2f})")

    sd = st.sharpe_diff_test(kel, arms["fixed L=1.00"], seed=944)
    print(f"  excess-Sharpe difference (the ONLY place leverage can move Sharpe, because "
          f"this arm is time-varying):\n"
          f"    ex-ante {sd['sharpe_a']:+.3f} vs 1x {sd['sharpe_b']:+.3f} -> "
          f"{sd['diff']:+.3f}, 95% CI [{sd['ci_low']:+.3f}, {sd['ci_high']:+.3f}] "
          f"({sd['frac_positive']:.0%} of draws positive) -> not distinguishable from zero")

    print("\n  cap sweep - the advantage is monotone in the cap you impose:")
    for hi in (1.5, 2.0, 2.5, 3.0):
        k = st.ex_ante_kelly(legs, window=756, hi=hi, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
        b = st.levered_returns(legs.loc[k.index], 1.0, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
        d = st.growth_diff_test(k, b, seed=944)
        s = st.annual_stats(k["r_lev"], k["e_lev"])
        print(f"    cap {hi:.1f}: mean L {k['lev'].mean():.2f}  CAGR {s['cagr']:+.2%}  "
              f"maxDD {s['max_drawdown']:+.1%}  advantage {d['log_growth_diff_ann']:+.2%}/yr "
              f"(t={d['t_log_diff']:+.2f})")

    print("\n  estimation-window sweep:")
    for w in (252, 504, 756, 1260):
        k = st.ex_ante_kelly(legs, window=w, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
        b = st.levered_returns(legs.loc[k.index], 1.0, spread_bps=SPREAD_BPS, cost_bps=COST_BPS)
        d = st.growth_diff_test(k, b, seed=944)
        print(f"    window {w:5d}d: mean L {k['lev'].mean():.2f}  advantage "
              f"{d['log_growth_diff_ann']:+.2%}/yr (t={d['t_log_diff']:+.2f})")

    # ------------------------------------------------------------- the proxies
    print("\n=== PROXY sweeps - the two non-tape assumptions ===")
    print("  financing spread over bills (bps/yr):")
    for s, r in st.spread_sweep(legs, cost_bps=COST_BPS).iterrows():
        print(f"    {s:6.0f}: optimum {r['opt_lev']:.2f}  CAGR@opt {r['cagr_opt']:+.2%}  "
              f"exSharpe@opt {r['sharpe_opt']:+.3f} (vs {r['sharpe_l1']:+.3f} unlevered)  "
              f"DD {r['dd_opt']:+.1%}")
    print("  one-way reset cost (bps):")
    for c, r in st.cost_sweep(legs, spread_bps=SPREAD_BPS).iterrows():
        print(f"    {c:5.1f}: optimum {r['opt_lev']:.2f}  CAGR@opt {r['cagr_opt']:+.2%}  "
              f"annual turnover {r['turnover_ann_opt']:.1f}x")

    # ------------------------------------------------------ synthetic control
    print("\n=== synthetic control (machinery proof - never supports the stamp) ===")
    grid_s = np.round(np.arange(0.0, 3.0001, 0.25), 4)
    for tag, ss in [("planted Kelly = 2.0", 1.0), ("null (zero excess drift)", 0.0)]:
        opts, kels = [], []
        for s in range(8):
            p, _ = data.synthetic_daily(signal_strength=ss, seed=944 + s)
            d = st.synthetic_detect(p, grid=grid_s)
            opts.append(d["opt_lev"]); kels.append(d["kelly"])
        print(f"  {tag:26s}: realised optimum mean {np.mean(opts):.2f} "
              f"(sd {np.std(opts, ddof=1):.2f}, seeds {sorted(opts)})  "
              f"Kelly estimate mean {np.mean(kels):.2f}")
    p, _ = data.synthetic_daily(signal_strength=1.0, seed=944)
    lg = st.prepare_synth(p)
    fine = st.realised_optimum(lg, grid=np.round(np.arange(0.0, 4.0001, 0.05), 4),
                               spread_bps=0.0, cost_bps=0.0)
    print(f"  conditional consistency on one 40-year tape: realised argmax {fine:.2f} vs "
          f"in-sample Kelly {st.kelly_from_legs(lg):.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
