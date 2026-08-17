"""Real-tape verification — Study 959 (Crypto Fee War). Regenerates docs/results.md.

Reads the cached daily closes of the ten US spot-bitcoin ETFs that launched together on
2024-01-11, plus BTC-USD (the 24/7 spot benchmark), BIL (cash) and Grayscale's later Mini
Trust (out-of-cohort cross-check). Measures each fund's realised tracking difference three
ways, tests whether the published fee ranking predicts the outcome ranking, prints the
cheapest-minus-priciest spread with a HAC t and a block-bootstrap CI, hunts the waiver
expiries, and costs the two ways to act on it. Network only on ``--fetch``.

    python studies/959-crypto-etf-fee-war/examples/verify.py            # cache-only
    python studies/959-crypto-etf-fee-war/examples/verify.py --fetch    # refresh the tapes
    python studies/959-crypto-etf-fee-war/examples/verify.py --sweep    # synthetic size/power
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fee_war import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px_all = data.load_prices()
    px = data.cohort_frame(px_all)          # cohort + BTC-USD, common window
    cohort = list(data.COHORT)

    print(f"cohort {', '.join(cohort)}  vs {data.BENCH}")
    print(f"window {px.index[0].date()} -> {px.index[-1].date()}  n={len(px)}  "
          f"fp={data.fingerprint(px)}   as-of {data.AS_OF}")
    print(f"excluded: {data.EXCLUDED}")

    print("\n=== the measurement floor (why the instrument is fund-vs-fund) ===")
    fl = st.measurement_floor(px, cohort, data.BENCH)
    print(f"  daily sd of a fund-minus-BTC difference : {fl['sd_daily_vs_bench_bp']:7.1f} bp")
    print(f"  daily sd of a fund-minus-fund difference: {fl['sd_daily_vs_peer_bp']:7.1f} bp   "
          f"(ratio {fl['ratio']:.1f}x)")
    print(f"  smallest |t|=2-detectable gap on {fl['n_months']} months: "
          f"vs BTC {fl['detectable_vs_bench_bpy']:.0f} bp/yr | vs a peer {fl['detectable_vs_peer_bpy']:.0f} bp/yr")

    print("\n=== realised tracking difference, three estimators (bp/yr) ===")
    tbl = st.tracking_table(px, cohort, data.BENCH, fees=data.FEE_BPS)
    print(f"  {'fund':6s} {'fee':>6s} {'endpoint':>9s} {'slope':>8s} {'t':>6s} "
          f"{'monthly':>9s} {'t':>6s} {'vs cohort':>10s} {'t':>6s}")
    for tk, r in tbl.iterrows():
        print(f"  {tk:6s} {r['fee_bps']:6.0f} {r['td_endpoint_bpy']:9.1f} {r['td_slope_bpy']:8.1f} "
              f"{r['t_slope']:6.2f} {r['td_monthly_bpy']:9.1f} {r['t_monthly']:6.2f} "
              f"{r['td_rel_bpy']:10.1f} {r['t_rel']:6.2f}")

    print("\n=== the headline: fee-cheapest minus fee-priciest (pure tape, no fee input) ===")
    cheap = min(cohort, key=lambda t: data.FEE_BPS[t])
    dear = max(cohort, key=lambda t: data.FEE_BPS[t])
    hd = st.pair_spread(px, cheap, dear)
    print(f"  {cheap} - {dear}: {hd['spread_bpy']:+.1f} bp/yr  HAC t={hd['tstat']:+.2f}  "
          f"positive in {hd['pos_months']}/{hd['n_months']} months")
    print(f"    block-bootstrap 95% CI [{hd['ci_low']:+.1f}, {hd['ci_high']:+.1f}] bp/yr, "
          f"share<0 {hd['frac_negative']:.4f}")
    print(f"    monthly sd {hd['sd_month_bp']:.1f} bp | daily sd {hd['sd_daily_bp']:.1f} bp")

    print("\n  every cheap fund against GBTC:")
    for tk in cohort:
        if tk == dear:
            continue
        p = st.pair_spread(px, tk, dear)
        print(f"    {tk:6s} {p['spread_bpy']:+8.1f} bp/yr  t={p['tstat']:+6.2f}  "
              f"{p['pos_months']:2d}/{p['n_months']} months positive")

    print("\n=== is the headline an artefact of its two anchors? ===")
    # The monthly mean telescopes, so its point estimate IS an endpoint estimate between the
    # first and last complete month-end. Test those anchors directly: trim months off both
    # ends, and re-estimate with every session as an anchor (the OLS trend slope).
    mm_hd = st.monthly_log_returns(px[[cheap, dear]])
    xh = (mm_hd[cheap] - mm_hd[dear]).dropna()
    for k in (1, 2, 3):
        xt = xh.iloc[k:len(xh) - k]
        print(f"  trim {k} month(s) off BOTH ends: n={len(xt)}  "
              f"{xt.mean() * st.MONTHS * 1e4:+7.1f} bp/yr  t={st.newey_west_t(xt.to_numpy(), lags=3):+5.2f}")
    sl_hd = st.td_trend_slope(px[cheap], px[dear])
    print(f"  all-{sl_hd['n']}-anchor OLS trend slope:  {sl_hd['slope_bpy']:+7.1f} bp/yr "
          f"(HAC t optimistic: residual acf1={sl_hd['resid_acf1']:.2f})")
    slopes = [st.td_trend_slope(px[t], px[dear])["slope_bpy"] for t in cohort if t != dear]
    print(f"  same slope on the nine cheap legs: median {np.median(slopes):+.1f} bp/yr "
          f"({min(slopes):+.1f} to {max(slopes):+.1f})")

    print("\n=== era cut (split 2025-01-01) ===")
    for tag, e in st.era_cut(px, cheap, dear).items():
        if e is None:
            continue
        print(f"  {tag:6s} n={e['n_months']:2d} months: {e['spread_bpy']:+8.1f} bp/yr  "
              f"t={e['tstat']:+6.2f}  {e['pos_months']}/{e['n_months']} positive")

    print("\n=== does the fee RANKING predict the tracking-difference ranking? ===")
    start, end = px.index[0], px.index[-1]
    blended = {tk: data.blended_fee_bps(tk, start, end) for tk in cohort}
    print("  waiver-blended effective fees (ASSUMPTION): "
          + ", ".join(f"{k} {v:.1f}" for k, v in blended.items()))
    for fee_tag, fee_map in [("headline fee", data.FEE_BPS), ("waiver-blended", blended)]:
        for sub_tag, sub in [("all ten", cohort), ("cheap nine (ex-GBTC)", [c for c in cohort if c != dear])]:
            fees = [fee_map[t] for t in sub]
            tds = [tbl.loc[t, "td_rel_bpy"] for t in sub]
            rt = st.rank_test(fees, tds)
            pt = st.pass_through(fees, tds)
            print(f"  {fee_tag:14s} | {sub_tag:20s}: Spearman {rt['spearman']:+.3f}  "
                  f"perm p={rt['p_perm']:.4f} (needs |rho|>{rt['crit_5pct']:.3f})  "
                  f"| TD = {pt['intercept']:+.1f} {pt['slope']:+.3f}*fee, R2={pt['r2']:.3f}")

    print("\n=== do the waiver expiries show up in the tape? ===")
    ev = st.waiver_event_test(px, cohort, data.WAIVER)
    for tk, r in ev.iterrows():
        if not np.isfinite(r["step_bpy"]):
            print(f"  {tk:6s} end {r['waiver_end']}: too few months either side (n_pre={r['n_pre']})")
            continue
        print(f"  {tk:6s} end {r['waiver_end']}: {r['pre_bpy']:+7.1f} -> {r['post_bpy']:+7.1f} bp/yr "
              f"= step {r['step_bpy']:+7.1f}  Welch t={r['welch_t']:+5.2f}  "
              f"(expected {-data.FEE_BPS[tk] + data.WAIVER[tk][0]:+.0f})")

    print("\n=== out-of-cohort cross-check: Grayscale's own Mini Trust (BTC, 15 bp) ===")
    mini = px_all[[data.MINI, dear, cheap]].dropna()
    for other in (dear, cheap):
        p = st.pair_spread(mini, data.MINI, other)
        print(f"  BTC - {other:5s} ({p['n_months']} months from {mini.index[0].date()}): "
              f"{p['spread_bpy']:+8.1f} bp/yr  t={p['tstat']:+6.2f}  "
              f"{p['pos_months']}/{p['n_months']} positive")

    print("\n=== the ownership race, excess-of-cash (2 bp one-way, one execution lag) ===")
    px_cash = px_all[cohort + [data.CASH]].dropna()
    race = st.rotation_race(px_cash, cohort, data.CASH, data.FEE_BPS)
    print(f"  cheapest={race['cheapest']}  priciest={race['priciest']}  "
          f"rotation switches={race['n_switches']}  n={race['n_days']}")
    for tag, a in race["arms"].items():
        print(f"    {tag:14s} excess Sharpe {a['excess_sharpe']:+.4f}  CAGR {a['cagr']:+.2%}  "
              f"vol {a['vol_ann']:.1%}  total {a['total_return']:+.2%}")
    print(f"  Sharpe gap cheapest - priciest: {race['sharpe_gap_cheap_minus_dear']:+.4f} "
          f"(a basis point a month is invisible at 50% vol)")

    print("\n=== acting on it (1): switch cost, a purchase decision ===")
    for row in st.switch_break_even(hd["spread_bpy"]):
        print(f"  one-way {row['one_way_bps']:5.1f} bp -> repaid in {row['break_even_days']:6.1f} days")

    print("\n=== acting on it (2): the taxable switch (tax rate + gain are ASSUMPTIONS) ===")
    for row in st.tax_break_even(hd["spread_bpy"]):
        print(f"  tax {row['tax_rate']:.3f}  embedded gain {row['embedded_gain']:+.0%}: "
              f"costs {row['cost_of_nav']:.2%} of NAV now -> repaid in {row['break_even_years']:5.1f} yrs")

    print("\n=== acting on it (3): the long/short, where the short pays borrow (ASSUMPTION) ===")
    for row in st.long_short_net(hd["spread_bpy"]):
        print(f"  borrow {row['borrow_bpy']:6.1f} bp/yr -> net {row['net_bpy']:+8.1f} bp/yr  "
              f"{'alive' if row['alive'] else 'DEAD'}")

    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    for ss, tag in [(1.0, "planted fee ladder"), (0.0, "null: one fee, dispersed fee sheet")]:
        p, tr = data.synthetic_panel(signal_strength=ss, seed=959)
        d = st.synthetic_detect(p, tr)
        print(f"  {tag:36s}: pair {d['pair_spread_bpy']:+7.1f} bp/yr (planted {d['planted_gap_bpy']:.0f}) "
              f"t={d['pair_t']:+6.2f} | Spearman {d['spearman']:+.3f} p={d['p_perm']:.3f} | "
              f"pass-through {d['pass_through_slope']:+.3f} R2={d['pass_through_r2']:.3f}")
    rhos, ps_, ts_ = [], [], []
    for s in range(8):
        p, tr = data.synthetic_panel(signal_strength=0.0, seed=959 + s)
        d = st.synthetic_detect(p, tr)
        rhos.append(d["spearman"]); ps_.append(d["p_perm"]); ts_.append(d["pair_t"])
    print(f"  null across 8 seeds: mean Spearman {np.mean(rhos):+.3f} (sd {np.std(rhos):.3f}), "
          f"max |pair t| {np.max(np.abs(ts_)):.2f}, fires at 5% on {int(np.sum(np.array(ps_) < 0.05))}/8")
    print("  (the full 120-seed size / 24-seed power sweep quoted in docs/results.md is "
          "reproduced by `--sweep`; it takes a few minutes)")


def sweep(n_null: int = 120, n_power: int = 24, n_perm: int = 2000) -> None:
    """The size-and-power sweep quoted in docs/results.md. Offline, fixed seeds, slow."""
    rhos, fires, ts_ = [], [], []
    for s in range(n_null):
        p, tr = data.synthetic_panel(signal_strength=0.0, seed=959 + s)
        d = st.synthetic_detect(p, tr, n_perm=n_perm)
        rhos.append(d["spearman"]); fires.append(d["p_perm"] < 0.05); ts_.append(abs(d["pair_t"]))
    print(f"NULL, {n_null} seeds : mean Spearman {np.mean(rhos):+.3f}, rank test fires at 5% on "
          f"{100 * np.mean(fires):.1f}% (nominal 5%), max |pair t| {max(ts_):.2f}")
    pt, sl, r2, fired = [], [], [], 0
    for s in range(n_power):
        p, tr = data.synthetic_panel(signal_strength=1.0, seed=959 + s)
        d = st.synthetic_detect(p, tr, n_perm=n_perm)
        pt.append(d["pair_t"]); sl.append(d["pass_through_slope"])
        r2.append(d["pass_through_r2"]); fired += int(d["p_perm"] < 0.05)
    print(f"POWER, {n_power} seeds: mean pair t {np.mean(pt):+.2f}, pass-through slope "
          f"{np.mean(sl):+.3f} +/- {np.std(sl, ddof=1):.3f}, R2 {np.mean(r2):.3f}, "
          f"rank test fires on {fired}/{n_power}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    else:
        main(fetch="--fetch" in sys.argv)
