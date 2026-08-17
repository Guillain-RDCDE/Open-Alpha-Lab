"""Real-tape verification — Study 946 (Distribution is not Return).

Regenerates every number in docs/results.md. Reads the two cached daily tapes (total-return
and price-only) for fifteen listed income funds plus SPY and BIL, reconstructs each fund's
trailing-12-month distribution rate, sorts the cross-section on it, and reports what that
rank predicts: next month's payout, next month's price return, next month's TOTAL return.

    python studies/946-distribution-rate-illusion/examples/verify.py            # cache-only
    python studies/946-distribution-rate-illusion/examples/verify.py --fetch    # refresh tapes

It also runs the offline synthetic control (planted world + null), which proves the
machinery is unbiased and never supports the real-tape stamp.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dist_illusion import data, strategy as st  # noqa: E402

SPLIT = "2020-06-30"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    tr = data.load_prices(kind="tr")
    pr = data.load_prices(kind="pr")
    panel = data.monthly_panel(tr, pr)
    legs = st.sorted_legs(panel)

    common = tr.index.intersection(pr.index)
    # The stamp fingerprints RETURNS, not levels: auto_adjust=True back-adjusts the whole
    # history on every re-fetch, so a level fingerprint drifts without a single return
    # changing. See data.returns_fingerprint.
    print(f"tapes: {common[0].date()} -> {common[-1].date()}  "
          f"fp(tr rets)={data.returns_fingerprint(tr.loc[common])}  "
          f"fp(pr rets)={data.returns_fingerprint(pr.loc[common])}")
    print(f"as-of {data.AS_OF}  |  {len(panel['funds'])} funds  |  guard {panel['guard']}")
    print(f"held months: {legs.index[0].date()} -> {legs.index[-1].date()}  n={len(legs)}  "
          f"cross-section {legs['n'].min()}-{legs['n'].max()} funds, k={legs['k'].min()}-{legs['k'].max()}")

    print("\n=== trailing distribution rate per fund (the ranking variable; a PROXY) ===")
    D = panel["dist_rate"]
    for f in panel["funds"]:
        s = D[f].dropna()
        tot = panel["total"][f].dropna()
        pri = panel["price"][f].dropna()
        if len(s) == 0:
            continue
        cagr_t = (1 + tot).prod() ** (12 / len(tot)) - 1
        cagr_p = (1 + pri).prod() ** (12 / len(pri)) - 1
        print(f"  {f:5s} n={len(tot):3d} mo from {tot.index[0].date()}  "
              f"dist rate mean {s.mean():6.2%} last {s.iloc[-1]:6.2%}  "
              f"total CAGR {cagr_t:+7.2%}  price CAGR {cagr_p:+7.2%}")

    print("\n=== Fama-MacBeth: what does the payout rank predict? (bps/month per 1 sd) ===")
    for target, label in (("dist", "next payout "), ("price", "price return"),
                          ("total", "TOTAL return")):
        fm = st.fama_macbeth(panel, target)
        print(f"  {label}: {fm['mean_bps']:+8.1f} bps  HAC t = {fm['tstat']:+6.2f}  n={fm['n_months']}")

    print("\n=== tercile sort: high payout minus low payout ===")
    print(f"  trailing payout at formation: high {legs['dhi'].mean():.2%}  low {legs['dlo'].mean():.2%}  "
          f"spread {legs['dhi'].mean() - legs['dlo'].mean():.2%}")
    for col, label in (("hml_d", "payout      "), ("hml_p", "price return"),
                       ("hml", "TOTAL return")):
        print(f"  {label}: {legs[col].mean() * 1e4:+8.1f} bps/mo  HAC t = {st.newey_west_t(legs[col].to_numpy()):+6.2f}")
    print(f"  give-back ratio (-hml_price / hml_payout): {st.giveback_ratio(legs):.2f}"
          f"  [1.00 = clean wash; the gap from 1.00 is exactly the TOTAL leg, so it is "
          f"as insignificant as it is]")
    ident = float((legs["hml"] - legs["hml_d"]).mean() * 1e4)
    corr = float(np.corrcoef(legs["hml_p"], legs["hml"] - legs["hml_d"])[0, 1])
    print(f"  IDENTITY CHECK: hml_price = hml_total - hml_payout -> "
          f"{ident:+.1f} bps vs {legs['hml_p'].mean() * 1e4:+.1f} bps, corr {corr:.5f}")
    print("    (so the price leg is arithmetic, not a second experiment: its t is the "
          "payout-persistence t carried over once the total leg contributes nothing)")

    print("\n=== bootstrap CIs (2,000 draws, 6-month blocks) ===")
    for col in ("hml_d", "hml_p", "hml"):
        ci = st.block_bootstrap_ci(legs[col])
        print(f"  {col:6s}: {ci['mean_bps']:+7.1f} bps  95% CI [{ci['ci_low']:+7.1f}, {ci['ci_high']:+7.1f}]  "
              f"P(>0) = {ci['frac_positive']:.3f}")
    ci_h = st.block_bootstrap_ci(legs["hml"])
    print(f"  the TOTAL-return null is BOUNDED, not precise: 95% CI = "
          f"[{ci_h['ci_low'] * 12 / 100:+.2f}%, {ci_h['ci_high'] * 12 / 100:+.2f}%] per year — "
          f"it rules out the marketed reading (payout rank -> better total return) above "
          f"~{ci_h['ci_high'] * 12 / 100:+.2f}%/yr, and rules out nothing on the downside")

    print("\n=== excess-of-cash race (both legs and SPY minus BIL) ===")
    r = st.race(panel, legs)
    for tag in ("hi", "lo", "bench"):
        s = r[tag]
        print(f"  {tag:5s}: exSharpe {s['sharpe']:+.3f}  mean {s['mean_bps']:+6.1f} bps  "
              f"vol {s['vol_ann']:.1%}  HAC t {s['tstat']:+.2f}  maxDD {s['max_drawdown']:+.1%}")
    for tag in ("hi_abs", "lo_abs", "bench_abs"):
        print(f"  {tag:9s}: total-return CAGR {r[tag]['cagr']:+.2%}  maxDD {r[tag]['max_drawdown']:+.1%}")
    print("\n=== CAPM control (excess-of-cash vs SPY excess-of-cash) ===")
    for tag in ("capm_hi", "capm_lo", "capm_hml"):
        c = r[tag]
        print(f"  {tag:9s}: alpha {c['alpha_bps']:+6.1f} bps/mo (HAC t {c['t_alpha']:+.2f})  "
              f"beta {c['beta']:+.3f}  R2 {c['r2']:.3f}")

    print(f"\n=== era cut (split {SPLIT}) ===")
    for tag, e in st.era_cut(panel, legs, split=SPLIT).items():
        if e is None:
            continue
        print(f"  {tag:5s} n={e['n_months']:3d} {e['start']} -> {e['end']}: "
              f"payout {e['hml_d_bps']:+6.1f}  price {e['hml_p_bps']:+7.1f} (t {e['t_hml_p']:+5.2f})  "
              f"TOTAL {e['hml_bps']:+7.1f} (t {e['t_hml']:+5.2f})  give-back {e['giveback']:.2f}  "
              f"alpha {e['alpha_bps']:+.1f} (t {e['t_alpha']:+.2f})")

    print("\n=== cost x borrow sweep on the high-minus-low leg ===")
    print(f"  one-way turnover/month: high leg {legs['to_hi'].mean():.3f}  low leg {legs['to_lo'].mean():.3f}")
    sweep = st.cost_borrow_sweep(legs)
    for _, row in sweep.iterrows():
        if (row["cost_bps"], row["borrow_bps"]) in ((0.0, 0.0), (5.0, 50.0), (5.0, 100.0),
                                                    (10.0, 100.0), (25.0, 200.0)):
            print(f"  cost {row['cost_bps']:5.1f} bps, borrow {row['borrow_bps']:5.1f} bps/yr: "
                  f"{row['mean_bps']:+7.1f} bps/mo (t = {row['tstat']:+.2f})")

    print("\n=== robustness ===")
    for tag, kw in (("tercile (frac 1/3)", dict(frac=1 / 3)),
                    ("quintile-ish (0.20)", dict(frac=0.20)),
                    ("wide sort (0.40)", dict(frac=0.40))):
        L = st.sorted_legs(panel, **kw)
        print(f"  {tag:20s}: price {L['hml_p'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml_p'].to_numpy()):+5.2f})  "
              f"TOTAL {L['hml'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml'].to_numpy()):+5.2f})")
    for tag, kw in (("no guard (live read)", dict(guard=None)), ("guard 0.40", dict(guard=0.40))):
        P2 = data.monthly_panel(tr, pr, **kw)
        L = st.sorted_legs(P2)
        print(f"  {tag:20s}: price {L['hml_p'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml_p'].to_numpy()):+5.2f})  "
              f"TOTAL {L['hml'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml'].to_numpy()):+5.2f})")
    P3 = data.monthly_panel(tr, pr, funds=[f for f in data.FUNDS if f != "NUSI"])
    L = st.sorted_legs(P3)
    print(f"  {'drop NUSI outright':20s}: price {L['hml_p'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml_p'].to_numpy()):+5.2f})  "
          f"TOTAL {L['hml'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L['hml'].to_numpy()):+5.2f})")
    P4 = data.monthly_panel(tr, pr, funds=list(data.CORE_FUNDS))
    L4 = st.sorted_legs(P4, min_funds=5)
    print(f"  {'option-income only':20s}: n={len(L4)} {L4.index[0].date()}->{L4.index[-1].date()}  "
          f"payout {L4['hml_d'].mean() * 1e4:+6.1f}  price {L4['hml_p'].mean() * 1e4:+7.1f} "
          f"(t {st.newey_west_t(L4['hml_p'].to_numpy()):+5.2f})  "
          f"TOTAL {L4['hml'].mean() * 1e4:+7.1f} (t {st.newey_west_t(L4['hml'].to_numpy()):+5.2f})")

    print("\n=== synthetic control (offline; machinery proof, never supports the stamp) ===")
    for tag, ss in (("null  (ss=0)", 0.0), ("planted (ss=1)", 1.0)):
        p, truth = data.synthetic_panel(signal_strength=ss, seed=946)
        d = st.synthetic_detect(p)
        print(f"  {tag}: FM total {d['fm_total_bps']:+7.1f} (t {d['t_total']:+6.2f})  "
              f"price {d['fm_price_bps']:+7.1f} (t {d['t_price']:+6.2f})  "
              f"payout {d['fm_dist_bps']:+6.1f} (t {d['t_dist']:+6.2f})  "
              f"give-back {d['giveback']:.2f}  [planted slope/sd {truth['planted_slope_per_sd'] * 1e4:+.1f} bps]")
    p, _ = data.synthetic_panel(signal_strength=0.0, beta_slope=0.5, seed=946)
    d = st.synthetic_detect(p)
    print(f"  beta confound (ss=0, beta_slope=0.5): raw HML {d['hml_bps']:+.1f} (t {d['t_hml']:+.2f})  ->  "
          f"CAPM alpha {d['capm_hml']['alpha_bps']:+.1f} (t {d['capm_hml']['t_alpha']:+.2f}), beta {d['capm_hml']['beta']:+.3f}")
    nulls = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=946 + s)[0])["fm_total_bps"]
        for s in range(8)
    ])
    print(f"  null across 8 seeds: FM total mean {nulls.mean():+.1f} bps, sd {nulls.std(ddof=1):.1f}, "
          f"|mean| > 20 bps on {(np.abs(nulls) > 20).sum()}/8")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main(fetch="--fetch" in sys.argv)
