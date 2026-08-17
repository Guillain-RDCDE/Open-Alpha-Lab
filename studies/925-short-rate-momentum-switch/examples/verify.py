"""Real-tape verification — Study 925 (Front-End Trend). Regenerates docs/results.md.

Reads the cached SHY / IEF / TLT / BIL total-return closes and the ^IRX 13-week bill
yield, builds the front-end trend signal (3-month change in ^IRX, one-day lag), and races
the resulting TLT/BIL switch against static IEF, static TLT and a frequency-matched random
control — every arm excess-of-cash. Prints the excess-Sharpe race, the HAC return-difference
*t*, bootstrap Sharpe and Sharpe-difference CIs, the seed-swept random control (gross and
net), the day-level conditional test, the era cut, the lookback sweep, the cost sweep, the
calendar-year table, the SHY 12-1 momentum cross-check and the synthetic control.

Network only on ``--fetch``.

    python studies/925-short-rate-momentum-switch/examples/verify.py            # cache-only
    python studies/925-short-rate-momentum-switch/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from front_end_trend import data, strategy as st  # noqa: E402

LOOKBACK = 63       # 3 months of trading days
COST_BPS = 2.0      # one-way x NAV, charged on every switch; no short leg -> no borrow
SPLIT = "2016-01-01"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices().dropna()
    irx, tlt, ief, bil, shy = px["^IRX"], px["TLT"], px["IEF"], px["BIL"], px["SHY"]

    print(f"SHY/IEF/TLT/BIL (total return) + ^IRX (yield, price-only): "
          f"{px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}  |  lookback {LOOKBACK}d  |  cost {COST_BPS:.0f} bps one-way")

    cmp = st.compare_rate_rule(irx, tlt, ief, bil, lookback=LOOKBACK, cost_bps=COST_BPS)

    print(f"\nin-duration fraction: {cmp['in_market_frac']:.1%}  |  "
          f"switches: {cmp['n_switches']} (~{cmp['n_switches']/19.1:.0f}/yr)")
    print("\n=== excess-of-cash race (all arms minus BIL total return) ===")
    for tag, key, dd in [("switch (rule)", "switch", "dd_switch_abs"),
                         ("static IEF", "static_mid", "dd_mid_abs"),
                         ("static TLT", "static_long", "dd_long_abs"),
                         ("random ctrl", "random", "dd_rand_abs")]:
        s = cmp[key]
        print(f"  {tag:13s}: exSharpe {s['sharpe']:+.3f}  exCAGR {s['cagr']*100:+.2f}%  "
              f"vol {s['vol_ann']*100:5.2f}%  HAC t {s['tstat']:+.2f}  "
              f"MaxDD(abs) {cmp[dd]*100:+.1f}%")

    print(f"\nexcess-Sharpe advantage (switch - static IEF): {cmp['excess_sharpe_adv']:+.3f}")
    print(f"  HAC t on daily return diff vs static IEF : {cmp['t_diff_vs_mid']:+.2f}")
    print(f"  HAC t on daily return diff vs static TLT : {cmp['t_diff_vs_long']:+.2f}")
    print(f"  advantage vs this one random seed        : {cmp['sharpe_adv_vs_random']:+.3f} "
          f"(t={cmp['t_diff_vs_random']:+.2f})")

    print("\n=== bootstrap CIs (excess-of-cash, 2000 draws, 21-day blocks) ===")
    for tag, key in [("switch", "e_switch"), ("static IEF", "e_mid"), ("static TLT", "e_long")]:
        ci = st.bootstrap_sharpe_ci(cmp[key])
        print(f"  {tag:11s}: Sharpe {ci['sharpe']:+.3f}  95% CI "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  frac<0 {ci['frac_negative']:.3f}")
    d = st.bootstrap_diff_ci(cmp["e_switch"], cmp["e_mid"])
    print(f"  Sharpe DIFFERENCE (switch - IEF): {d['diff']:+.3f}  95% CI "
          f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  frac<0 {d['frac_negative']:.3f}")

    print("\n=== random control, swept over 30 seeds (one seed is one draw) ===")
    sig = st.rate_trend_signal(irx, lookback=LOOKBACK)
    for c in (COST_BPS, 0.0):
        sw = st.random_control_sweep(tlt, ief, bil, sig, n_seeds=30, cost_bps=c)
        print(f"  cost {c:4.1f} bps: adv mean {sw['adv_mean']:+.3f} (sd {sw['adv_sd']:.3f}, "
              f"range {sw['adv_min']:+.3f}..{sw['adv_max']:+.3f})  "
              f"t mean {sw['t_mean']:+.2f}  share t>=2: {sw['frac_t_ge_2']:.0%}")

    print("\n=== matched constant-weight blend (turnover-free exposure control) ===")
    for c in (COST_BPS, 0.0):
        r = st.matched_blend_race(tlt, ief, bil, sig, cost_bps=c)
        print(f"  cost {c:4.1f} bps: switch {r['sharpe_switch']:+.3f} vs blend "
              f"{r['sharpe_blend']:+.3f} (w={r['weight']:.3f}, turnover "
              f"{r['turnover_ann_blend']*100:.0f}%/yr)  adv {r['adv_vs_blend']:+.3f} "
              f"(return-diff t {r['t_diff_vs_blend']:+.2f})")

    print("\n=== day-level conditional test (gross, no portfolio construction) ===")
    cd = st.conditional_day_test(irx, tlt, bil, lookback=LOOKBACK)
    print(f"  excess TLT on  'own duration' days: {cd['on_bp']:+.2f} bp/d  "
          f"(n={cd['n_on']}, HAC t {cd['t_on']:+.2f})")
    print(f"  excess TLT on  'sit in bills' days: {cd['off_bp']:+.2f} bp/d  "
          f"(n={cd['n_off']}, HAC t {cd['t_off']:+.2f})")
    print(f"  spread {cd['spread_bp']:+.2f} bp/d, HAC t {cd['spread_hac_t']:+.2f} "
          f"(Welch t {cd['welch_t']:+.2f}, iid — reference only)")

    print(f"\n=== era cut (split {SPLIT}) ===")
    for tag, e in st.era_cut(irx, tlt, ief, bil, split=SPLIT,
                             lookback=LOOKBACK, cost_bps=COST_BPS).items():
        if e is None:
            continue
        print(f"  {tag:5s} (n={e['n_days']}): IEF {e['sharpe_mid']:+.2f} / "
              f"switch {e['sharpe_switch']:+.2f}  adv {e['excess_sharpe_adv']:+.3f} "
              f"(t={e['t_diff_vs_mid']:+.2f})  in-duration {e['in_market_frac']:.0%}  "
              f"DD {e['dd_mid_abs']*100:+.1f}% -> {e['dd_switch_abs']*100:+.1f}%")

    print("\n=== lookback sweep (is 63 days special?) ===")
    for r in st.lookback_sweep(irx, tlt, ief, bil, cost_bps=COST_BPS):
        print(f"  {r['lookback']:3d}d: switch {r['sharpe_switch']:+.3f}  "
              f"adv {r['excess_sharpe_adv']:+.3f} (t={r['t_diff_vs_mid']:+.2f})  "
              f"in-duration {r['in_market_frac']:.0%}  switches {r['n_switches']}")

    print("\n=== cost sweep (no short leg -> no borrow; switch friction only) ===")
    for r in st.timer_stats(irx, tlt, ief, bil, lookback=LOOKBACK):
        print(f"  {r['cost_bps']:5.1f} bps: switch {r['sharpe_switch']:+.3f}  "
              f"adv {r['excess_sharpe_adv']:+.3f} (t={r['t_diff_vs_mid']:+.2f})  "
              f"MaxDD {r['dd_switch_abs']*100:+.1f}%")

    print("\n=== calendar years (absolute total return, %) — the FULL table, no selection ===")
    tbl = st.calendar_year_table(irx, tlt, ief, bil, lookback=LOOKBACK, cost_bps=COST_BPS)
    gap = tbl["switch_pct"] - tbl["ief_pct"]
    for y, row in tbl.iterrows():
        print(f"  {y}: switch {row['switch_pct']:+6.1f}%  IEF {row['ief_pct']:+6.1f}%  "
              f"TLT {row['tlt_pct']:+6.1f}%  gap {row['switch_pct']-row['ief_pct']:+6.1f}%  "
              f"in-duration {row['in_duration_frac']:.0%}")
    print(f"  years beating IEF: {int((gap > 0).sum())}/{len(gap)}  |  "
          f"mean gap {gap.mean():+.2f} pp/yr  |  median {gap.median():+.2f} pp/yr")

    # The two years the folklore remembers, measured on the tape rather than asserted.
    bt_y = st.run_backtest(tlt, bil, sig, cost_bps=COST_BPS)
    on23 = bt_y.loc["2023", "signal"]
    on23 = on23[on23 == 1.0]
    i0, i1 = on23.index[0], on23.index[-1]
    tlt_win = float(tlt.loc[i1] / tlt.loc[i0] - 1.0) * 100
    strat_win = float((1.0 + bt_y.loc[i0:i1, "r_strategy"]).prod() - 1.0) * 100
    print(f"  2022: {int(bt_y.loc['2022', 'signal'].sum())} of "
          f"{len(bt_y.loc['2022'])} sessions in duration")
    print(f"  2023: {len(on23)} sessions in duration ({i0.date()} -> {i1.date()}); "
          f"TLT over that window {tlt_win:+.1f}%, the rule's own compounded return "
          f"{strat_win:+.1f}%")

    print("\n=== cross-check: SHY 12-minus-1-month momentum instead of the ^IRX change ===")
    sig_shy = st.shy_momentum_signal(shy)
    c2 = st.compare(tlt, ief, bil, sig_shy, cost_bps=COST_BPS)
    print(f"  switch {c2['switch']['sharpe']:+.3f} vs IEF {c2['static_mid']['sharpe']:+.3f}  "
          f"adv {c2['excess_sharpe_adv']:+.3f} (t={c2['t_diff_vs_mid']:+.2f})  "
          f"in-duration {c2['in_market_frac']:.0%}  switches {c2['n_switches']}")
    sw2 = st.random_control_sweep(tlt, ief, bil, sig_shy, n_seeds=30, cost_bps=0.0)
    print(f"  gross vs random (30 seeds): adv {sw2['adv_mean']:+.3f}  t mean {sw2['t_mean']:+.2f}")
    # the fair control for a rule that is long duration 85% of the time
    for c in (COST_BPS, 0.0):
        rb = st.matched_blend_race(tlt, ief, bil, sig_shy, cost_bps=c)
        print(f"  vs matched blend, cost {c:4.1f} bps: switch {rb['sharpe_switch']:+.3f} vs "
              f"blend {rb['sharpe_blend']:+.3f} (w={rb['weight']:.3f})  "
              f"adv {rb['adv_vs_blend']:+.3f} (t {rb['t_diff_vs_blend']:+.2f})")
    for tag, sub in [("early  ", px.loc[:SPLIT]), ("late   ", px.loc[SPLIT:]),
                     ("ex-2022", px[px.index.year != 2022])]:
        rb = st.matched_blend_race(sub["TLT"], sub["IEF"], sub["BIL"],
                                   sig_shy.reindex(sub.index), cost_bps=COST_BPS)
        print(f"    {tag} (n={rb['n_days']}, w={rb['weight']:.2f}): "
              f"adv vs blend {rb['adv_vs_blend']:+.3f} (t {rb['t_diff_vs_blend']:+.2f})")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    f_pl = data.synthetic_daily(signal_strength=1.0, seed=925)[0]
    pl = st.synthetic_detect(f_pl)
    print(f"  planted rate-trend world: adv {pl['excess_sharpe_adv']:+.3f} "
          f"(t={pl['t_diff_vs_mid']:+.2f})  -> the detector fires")
    mb = st.matched_blend_race(f_pl["long"], f_pl["mid"], f_pl["cash"],
                               st.rate_trend_signal(f_pl["rate"], lookback=LOOKBACK),
                               cost_bps=COST_BPS)
    print(f"  ... and vs its own matched blend: adv {mb['adv_vs_blend']:+.3f} "
          f"(t {mb['t_diff_vs_blend']:+.2f}) -> the blend control has power too")
    nl = np.array([st.synthetic_detect(f)["excess_sharpe_adv"]
                   for f, _ in data.synthetic_panel(n_seeds=10, signal_strength=0.0)])
    pp = np.array([st.synthetic_detect(f)["excess_sharpe_adv"]
                   for f, _ in data.synthetic_panel(n_seeds=10, signal_strength=1.0)])
    print(f"  null  x10: adv mean {nl.mean():+.3f} (sd {nl.std(ddof=1):.3f}), "
          f"|adv|>=0.4 in {(np.abs(nl) >= 0.4).sum()}/10")
    print(f"  planted x10: adv mean {pp.mean():+.3f}, adv>=0.4 in {(pp >= 0.4).sum()}/10")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
