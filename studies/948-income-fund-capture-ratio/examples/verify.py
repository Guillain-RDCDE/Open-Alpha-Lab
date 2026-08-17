"""Real-tape verification — Study 948 (Capture Ratio). Regenerates docs/results.md.

Reads cached daily total-return closes for fourteen US income ETFs, their marketed
benchmarks (SPY / QQQ / IWM) and the cash leg (BIL), resamples to monthly total returns,
and prints: the per-fund capture scorecard with block-bootstrap CIs, the Henriksson-Merton
convexity with HAC *t* against a Bonferroni bar, plain CAPM alpha/beta, the excess-of-cash
Sharpe race, the era cut and rank-persistence test, the benchmark-assignment sweep, and the
cost × borrow sweep of the beta-neutral traded spread. Network only on ``--fetch``.

    python studies/948-income-fund-capture-ratio/examples/verify.py            # cache-only
    python studies/948-income-fund-capture-ratio/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from capture_ratio import data, strategy as st  # noqa: E402

SPLIT = "2021-01-01"
WINDOW = 36


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    raw = data.load_prices(repair_splits=False)
    px = data.load_prices()
    rets = data.to_monthly_returns(px)
    rets_raw = data.to_monthly_returns(raw)

    print(f"as-of {data.AS_OF}  |  {len(data.FUNDS)} income ETFs vs SPY/QQQ/IWM, cash = BIL")
    print(f"monthly total returns  n_months(frame)={len(rets.dropna(how='all'))}  "
          f"fp={data.fingerprint(rets.fillna(0.0))}")

    print("\n=== unadjusted-corporate-action screen (vendor 'adjusted' tape) ===")
    events = data.split_artifacts(raw)
    if not events:
        print("  no share-count events found in the raw tape")
    for ev in events:
        print(f"  {ev['ticker']:6s} {ev['date'].date()}  one-day ratio {ev['ratio']:.6f}  "
              f"-> simple factor {ev['factor']:.4g}  repaired={ev['repairable']}")
    for ev in events:
        tk = ev["ticker"]
        if tk not in data.FUND_BENCH:
            continue
        b = data.FUND_BENCH[tk]
        c_raw = st.capture_ratios(rets_raw[tk], rets_raw[b])
        c_fix = st.capture_ratios(rets[tk], rets[b])
        a_raw = st.capm_alpha(rets_raw[tk] - rets_raw["BIL"], rets_raw[b] - rets_raw["BIL"])
        a_fix = st.capm_alpha(rets[tk] - rets["BIL"], rets[b] - rets["BIL"])
        print(f"  {tk} raw    : down-capture {c_raw['down']:+.3f}  spread {c_raw['spread']:+.3f}"
              f"  alpha {a_raw['alpha_bps']:+.1f} bps/mo (t={a_raw['t_alpha']:+.2f})")
        print(f"  {tk} repaired: down-capture {c_fix['down']:+.3f}  spread {c_fix['spread']:+.3f}"
              f"  alpha {a_fix['alpha_bps']:+.1f} bps/mo (t={a_fix['t_alpha']:+.2f})")
    print("  (everything below runs on the REPAIRED tape)")

    sc = st.scorecard(rets, data.FUND_BENCH)
    zb = st.bonferroni_z(len(sc))
    print(f"\nBonferroni bar for {len(sc)} simultaneous tests (alpha=0.05): |t| >= {zb:.2f}")

    print("\n=== capture scorecard (monthly total returns, benchmark-sign partition) ===")
    print(f"{'fund':6s}{'bench':6s}{'n':>5s}{'from':>10s}  {'up':>6s}{'down':>7s}"
          f"{'spread':>8s}{'95% CI':>18s}{'P(>0)':>7s}{'conv':>7s}{'t':>7s}")
    for f, r in sc.iterrows():
        print(f"{f:6s}{r['bench']:6s}{int(r['n_months']):5d}{r['start'][:7]:>10s}  "
              f"{r['up']:6.3f}{r['down']:7.3f}{r['spread']:+8.3f}"
              f"   [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]".ljust(0)
              + f"{r['frac_positive']:7.2f}{r['convexity']:+7.2f}{r['t_convexity']:+7.2f}")

    pos = sc[(sc["t_convexity"] >= 2.0)]
    neg = sc[(sc["t_convexity"] <= -2.0)]
    passing = sc[sc["t_convexity"] >= zb]
    print(f"\nfunds with convexity t >= +2 (nominal): {len(pos)}  -> {list(pos.index)}")
    print(f"funds clearing the Bonferroni bar (+{zb:.2f}): {len(passing)} -> {list(passing.index)}")
    print(f"funds with convexity t <= -2 (concave):  {len(neg)}  -> {list(neg.index)}")
    print(f"cross-sectional spread: median {sc['spread'].median():+.3f}  "
          f"mean {sc['spread'].mean():+.3f}  min {sc['spread'].min():+.3f} "
          f"max {sc['spread'].max():+.3f}")
    print(f"bootstrap CI covers zero for {int(((sc['ci_low'] <= 0) & (sc['ci_high'] >= 0)).sum())}"
          f"/{len(sc)} funds")

    print("\n=== arithmetic vs geometric capture convention ===")
    for f, r in sc.iterrows():
        print(f"  {f:6s} arithmetic {r['spread']:+.3f}   geometric {r['geo_spread']:+.3f}")

    print("\n=== is it just beta? CAPM alpha (excess-of-cash, HAC L=6) ===")
    for f, r in sc.iterrows():
        print(f"  {f:6s} beta {r['beta']:.2f}  alpha {r['alpha_bps']:+7.1f} bps/mo "
              f"(t={r['t_alpha']:+.2f})  kink-intercept {r['kink_intercept_bps']:+7.1f} bps/mo")
    print(f"  max |t_alpha| across the panel: {sc['t_alpha'].abs().max():.2f}")

    print("\n=== excess-of-cash Sharpe race (fund vs its own benchmark) ===")
    for f, r in sc.iterrows():
        print(f"  {f:6s} fund {r['sharpe_fund']:+.3f}  bench {r['sharpe_bench']:+.3f}  "
              f"gap {r['sharpe_gap']:+.3f} (HAC t on monthly diff {r['t_sharpe_gap']:+.2f})  "
              f"DD {r['dd_fund']:+.1%} vs {r['dd_bench']:+.1%}")
    print(f"  funds beating their benchmark on excess Sharpe: "
          f"{int((sc['sharpe_gap'] > 0).sum())}/{len(sc)}")

    print(f"\n=== era cut (split {SPLIT}, >= 24 months each side) + rank persistence ===")
    era = st.era_spreads(rets, data.FUND_BENCH, split=SPLIT)
    for f, r in era.iterrows():
        print(f"  {f:6s} early(n={int(r['n_early'])}) {r['spread_early']:+.3f}   "
              f"late(n={int(r['n_late'])}) {r['spread_late']:+.3f}")
    rp = st.rank_persistence(era)
    print(f"  Spearman rho = {rp['rho']:+.3f} (p = {rp['p']:.3f}, n = {rp['n']})  "
          f"sign agreement {rp['sign_agreement']:.0%}")

    print("\n=== ASSUMPTION sweep: force SPY as the benchmark for every fund ===")
    sc_spy = st.scorecard(rets, data.FUND_BENCH, n_boot=800, force_bench="SPY")
    print(f"  median spread {sc_spy['spread'].median():+.3f}  "
          f"funds with t_convexity >= +2: {int((sc_spy['t_convexity'] >= 2).sum())}  "
          f"<= -2: {int((sc_spy['t_convexity'] <= -2).sum())}")
    print("  per-fund spread (mapped -> forced-SPY):")
    for f in sc.index:
        if f in sc_spy.index:
            print(f"    {f:6s} {sc['spread'][f]:+.3f} -> {sc_spy['spread'][f]:+.3f}")

    print("\n=== beta-neutral traded spread: cost x borrow sweep (36m rolling beta, t+1) ===")
    for f, b in data.FUND_BENCH.items():
        rows = st.traded_sweep(rets[f], rets[b], rets["BIL"], window=WINDOW,
                               cost_grid=(0.0, 5.0), borrow_grid=(0.0, 100.0))
        if not rows:
            print(f"  {f:6s} (too short for a {WINDOW}-month rolling beta)")
            continue
        g = [r for r in rows if r["cost_bps"] == 0 and r["borrow_bps"] == 0][0]
        n = [r for r in rows if r["cost_bps"] == 5 and r["borrow_bps"] == 100][0]
        print(f"  {f:6s} n={g['n_months']:3d}  gross {g['mean_bps']:+6.1f} bps/mo (t={g['t']:+.2f})"
              f"   net(5bps, 100bps borrow) {n['mean_bps']:+6.1f} (t={n['t']:+.2f})")

    print("\n=== borrow sweep on the best traded spread with >= 60 months of history ===")
    cands, all_cands = [], []
    for f, b in data.FUND_BENCH.items():
        r = st.traded_beta_neutral(rets[f], rets[b], rets["BIL"], window=WINDOW,
                                   cost_bps=0.0, borrow_bps_ann=0.0)
        if len(r) >= 12:
            all_cands.append((float(r.mean()), f, st.newey_west_t(r.to_numpy()), len(r)))
        if len(r) >= 60:
            cands.append((float(r.mean()), f))
    top = max(all_cands)
    print(f"  (best gross ignoring the 60-month floor: {top[1]} {top[0] * 1e4:+.1f} bps/mo, "
          f"t={top[2]:+.2f}, n={top[3]} — still short of |t| >= 2)")
    best = max(cands)[1]
    print(f"  best gross candidate: {best} ({max(cands)[0] * 1e4:+.1f} bps/mo gross)")
    for row in st.traded_sweep(rets[best], rets[data.FUND_BENCH[best]], rets["BIL"],
                               window=WINDOW, cost_grid=(5.0,),
                               borrow_grid=(0.0, 50.0, 100.0, 200.0)):
        print(f"  {best}: borrow {row['borrow_bps']:5.0f} bps -> {row['mean_bps']:+6.1f} bps/mo "
              f"(t={row['t']:+.2f}, Sharpe {row['sharpe']:+.2f})")

    print("\n=== the estimator's own null bias (20 zero-truth synthetic panels) ===")
    nb = st.null_spread_distribution(
        data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(20))
    print(f"  true capture spread = 0.000 for all {nb['n_obs']} fund-panels")
    print(f"  measured spread    : mean {nb['spread_mean']:+.3f}  median {nb['spread_median']:+.3f}"
          f"  sd {nb['spread_sd']:.3f}  positive {nb['spread_frac_positive']:.0%} of the time")
    print(f"  HM convexity coeff : mean {nb['convexity_mean']:+.3f}  "
          f"median {nb['convexity_median']:+.3f}  sd {nb['convexity_sd']:.3f}  (unbiased)")
    print("  NB: this is the UNCONDITIONAL bias (benchmark path random). It is a property")
    print("      of the estimator, and it must NOT be subtracted from a real-tape spread,")
    print("      which is computed on ONE realised benchmark path. For that, see below.")

    print("\n=== fund-matched null on the REAL tape (benchmark & cash paths held fixed) ===")
    mn0 = st.matched_null_panel(rets, data.FUND_BENCH, n_sims=1000)
    mn1 = st.matched_null_panel(rets, data.FUND_BENCH, n_sims=1000, include_alpha=True)
    print(f"{'fund':6s}{'observed':>10s}{'null(b)':>9s}{'excess':>9s}{'p':>7s}"
          f"{'null(a,b)':>11s}{'excess':>9s}")
    for f in mn0.index:
        print(f"{f:6s}{mn0['observed'][f]:+10.3f}{mn0['null_mean'][f]:+9.3f}"
              f"{mn0['excess_spread'][f]:+9.3f}{mn0['p_value'][f]:7.3f}"
              f"{mn1['null_mean'][f]:+11.3f}{mn1['excess_spread'][f]:+9.3f}")
    print(f"  null(b)   = zero-alpha, zero-convexity twin (a plain scaled index position).")
    print(f"            funds whose spread reaches p < 0.05 against it: "
          f"{int((mn0['p_value'] < 0.05).sum())}/{len(mn0)}  (min p = {mn0['p_value'].min():.3f})")
    print(f"            median excess spread vs that twin: {mn0['excess_spread'].median():+.3f}")
    print(f"  null(a,b) = the fund's OWN fitted alpha and beta, still zero convexity.")
    print(f"            it reproduces the observed spread to a median |error| of "
          f"{mn1['excess_spread'].abs().median():.3f} (max {mn1['excess_spread'].abs().max():.3f})")
    print("            -> the capture spread carries no information beyond (alpha, beta).")
    print("  CAVEAT: null(b) also fires on a perfectly linear fund with a positive alpha,")
    print("          so a small p there means 'beats a scaled index', not 'is convex'.")
    print("          Convexity as such is the Henriksson-Merton column above.")

    print("\n=== synthetic control (machinery proof only — never supports the stamp) ===")
    planted, tr_p = data.synthetic_panel(signal_strength=1.0, seed=948)
    null, tr_n = data.synthetic_panel(signal_strength=0.0, seed=948)
    dp = st.synthetic_detect(planted, tr_p)
    dn = st.synthetic_detect(null, tr_n)
    print(f"  planted: corr(measured, true spread) = {dp['corr_measured_true']:+.3f}, "
          f"max |error| = {dp['max_abs_err']:.3f}, "
          f"{dp['n_hits_positive']} positive / {dp['n_hits_negative']} negative hits "
          f"of {dp['n_funds']}")
    fires = []
    for s in range(8):
        z, tz = data.synthetic_panel(signal_strength=0.0, seed=948 + s)
        d = st.synthetic_detect(z, tz, n_boot=200)
        fires.append(d["n_hits_positive"] + d["n_hits_negative"])
    print(f"  null x8 seeds: |t| >= 2 fires on {sum(fires)}/{8 * dn['n_funds']} fund-tests "
          f"(nominal 5% would give ~{0.05 * 8 * dn['n_funds']:.1f})")
    print(f"  null median measured spread {float(np.median(dn['table']['spread'])):+.3f} "
          f"(planted truth: 0.000 for every fund)")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main(fetch="--fetch" in sys.argv)
