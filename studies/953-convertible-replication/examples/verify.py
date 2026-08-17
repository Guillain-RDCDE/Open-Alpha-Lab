"""Real-tape verification — Study 953 (Replicating the Convert). Regenerates docs/results.md.

Fits CWB's daily excess-of-cash total return to a static long-only mix of SPY / QQQ / LQD
(the unspent weight in BIL) on the in-sample window only, freezes the weights, and scores
the out-of-sample residual alpha (HAC *t* + block-bootstrap CI), the convexity smile, an
era cut, a cost/fee sweep, an annual-refit variant and an ICVT cross-check. Network only
on ``--fetch``.

    python studies/953-convertible-replication/examples/verify.py            # cache-only
    python studies/953-convertible-replication/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from convert_repl import data, strategy as st  # noqa: E402

FACTORS = ("SPY", "QQQ", "LQD")
CASH = "BIL"
COST_BPS = 3.0
FEE_ANN = 0.0010


def run_fund(px: pd.DataFrame, fund: str, is_end: str) -> dict:
    sub = px[[fund, *FACTORS, CASH]].dropna()
    res = st.holdout(sub, fund, FACTORS, CASH, is_end, cost_bps=COST_BPS, fee_ann=FEE_ANN)
    rets = sub.pct_change().dropna()
    cx = st.convexity_test(res["ex_fund"], res["replica"], rets["SPY"] - rets[CASH],
                           mask=res["oos_mask"])
    return {"sub": sub, "res": res, "cx": cx}


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()

    out = run_fund(px, "CWB", data.IS_END)
    sub, res, cx = out["sub"], out["res"], out["cx"]
    fit = res["fit"]

    print(f"CWB / {'+'.join(FACTORS)} / {CASH}: {sub.index[0].date()} -> {sub.index[-1].date()}  "
          f"n={len(sub)}  fp={data.fingerprint(sub)}")
    print(f"as-of {data.AS_OF}  |  in-sample through {fit['is_end']}  |  "
          f"cost {COST_BPS:.0f} bps one-way, replica fee {FEE_ANN*1e4:.0f} bps/yr")

    print("\n=== the fitted replication (in-sample only, long-only, budget <= 1) ===")
    for k, v in fit["weights"].items():
        print(f"  {k:4s} {v:6.1%}")
    print(f"  cash {fit['w_cash']:6.1%}   (weights sum {1 - fit['w_cash']:.1%})")
    print(f"  in-sample R^2 {fit['r2_is']:.3f}  tracking error {fit['te_is_ann']:.2%}/yr  "
          f"intercept {fit['alpha_is_ann']:+.2%}/yr  n={fit['n_is']}")

    print("\n=== the hold-out - is the residual alpha distinguishable from zero? ===")
    for tag in ("is", "oos"):
        s = res[tag]
        print(f"  {tag.upper():4s} {s['start']} -> {s['end']}  n={s['n_days']}  "
              f"alpha {s['alpha_ann']:+.2%}/yr  HAC t {s['t_hac']:+.2f}  "
              f"TE {s['te_ann']:.2%}/yr  corr {s['corr']:.3f}")
    o = res["oos"]
    print(f"  OOS excess Sharpe: fund {o['sharpe_fund']:+.3f} vs replica "
          f"{o['sharpe_replica']:+.3f}  (gap {o['sharpe_gap']:+.3f})")
    print(f"  OOS vol: fund {o['vol_fund']:.2%} vs replica {o['vol_replica']:.2%}  |  "
          f"max DD: fund {o['dd_fund']:+.2%} vs replica {o['dd_replica']:+.2%}")

    m = res["oos_mask"]
    f, r = res["ex_fund"][m], res["replica"][m]
    k = float(f.std(ddof=1) / r.std(ddof=1))
    print(f"  vol-matched replica (x{k:.3f}): max DD {st.max_drawdown(r * k):+.2%} "
          f"vs fund {st.max_drawdown(f):+.2%}")
    fm, rm = st.to_monthly(f), st.to_monthly(r)
    print(f"  monthly R^2 {np.corrcoef(fm, rm)[0, 1] ** 2:.3f}  "
          f"monthly TE {float((fm - rm).std(ddof=1) * np.sqrt(12)):.2%}/yr")

    ci = st.block_bootstrap_mean_ci(res["diff"][m], seed=953)
    print(f"  block-bootstrap alpha CI (2,000 draws, 21-day blocks): "
          f"{ci['mean_ann']:+.2%} [{ci['ci_low']:+.2%}, {ci['ci_high']:+.2%}]  "
          f"share>0 {ci['frac_positive']:.1%}")

    print("\n=== the convexity claim (out-of-sample months, sorted by SPY excess) ===")
    print(f"  n months {cx['n_months']}  |  fund-minus-replica by tercile (bps/month):")
    print(f"    bottom {cx['mean_low_bps']:+7.1f} (t {cx['t_low']:+.2f})   "
          f"middle {cx['mean_mid_bps']:+7.1f}   top {cx['mean_high_bps']:+7.1f} "
          f"(t {cx['t_high']:+.2f})")
    print(f"  smile (tails - middle) {cx['smile_bps']:+.1f} bps/month  (Welch t {cx['t_smile']:+.2f})")
    print(f"  up-capture vs replica {cx['up_capture']:.3f}  down-capture {cx['down_capture']:.3f}  "
          f"asymmetry {cx['capture_asymmetry']:+.3f}")
    print(f"  Treynor-Mazuy curvature on the difference: gamma {cx['tm_gamma']:+.3f} "
          f"(HAC t {cx['t_tm_gamma']:+.2f})")

    print("\n=== era cut of the hold-out (split 2022-01-01) ===")
    for tag, e in st.era_cut(res).items():
        if e:
            print(f"  {tag:5s} {e['start']} -> {e['end']}  n={e['n_days']}  "
                  f"alpha {e['alpha_ann']:+.2%}/yr  HAC t {e['t_hac']:+.2f}")

    print("\n=== cost / fee sweep (out-of-sample alpha) ===")
    for row in st.cost_sweep(sub, "CWB", FACTORS, CASH, data.IS_END):
        print(f"  cost {row['cost_bps']:4.1f} bps one-way, replica fee {row['fee_ann_bps']:4.0f} bps/yr"
              f"  ->  alpha {row['alpha_ann']:+.2%}/yr  (t {row['t_hac']:+.2f})")

    print("\n=== robustness ===")
    unc = st.holdout(sub, "CWB", FACTORS, CASH, data.IS_END, nonneg=False,
                     cost_bps=COST_BPS, fee_ann=FEE_ANN)
    uw = unc["fit"]["weights"]
    print(f"  unconstrained fit: " + "  ".join(f"{a} {b:+.3f}" for a, b in uw.items())
          + f"  cash {unc['fit']['w_cash']:+.3f}  ->  alpha {unc['oos']['alpha_ann']:+.2%}/yr "
            f"(t {unc['oos']['t_hac']:+.2f})")
    print("    (the long-only budget never binds: the unconstrained solution is already inside it,"
          " so no short leg and no borrow cost arises)")
    print("  factor-set sweep (was the QQQ leg doing the work? a hindsight-flavoured choice):")
    for row in st.factor_set_sweep(sub, "CWB",
                                   [("SPY",), ("SPY", "LQD"), ("SPY", "QQQ"),
                                    ("SPY", "QQQ", "LQD")],
                                   CASH, data.IS_END, cost_bps=COST_BPS, fee_ann=FEE_ANN):
        print(f"    {row['factors']:12s} IS R^2 {row['r2_is']:.3f}  cash {row['w_cash']:5.1%}"
              f"  ->  alpha {row['alpha_ann']:+.2%}/yr  (t {row['t_hac']:+.2f})")
    print("  in-sample boundary sweep (was 2017-12-31 doing the work?):")
    for row in st.split_sweep(sub, "CWB", FACTORS, CASH,
                              [f"{y}-12-31" for y in range(2014, 2021)],
                              cost_bps=COST_BPS, fee_ann=FEE_ANN):
        print(f"    IS through {row['is_end']}  n_oos={row['n_oos']:5d}  ->  "
              f"alpha {row['alpha_ann']:+.2%}/yr  (t {row['t_hac']:+.2f})")
    rr = st.rolling_refit(sub, "CWB", FACTORS, CASH, "2013-12-31")
    print(f"  annual refit {rr['start']} -> {rr['end']}  n={rr['n_days']}  "
          f"alpha {rr['alpha_ann']:+.2%}/yr  (t {rr['t_hac']:+.2f})  TE {rr['te_ann']:.2%}/yr")
    print(f"    last fitted weights: " +
          "  ".join(f"{a} {b:.1%}" for a, b in rr["weights_path"][-1].items() if a != "through"))

    print("\n=== ICVT cross-check (a second convertible ETF, IS through 2020-12-31) ===")
    out2 = run_fund(px, "ICVT", "2020-12-31")
    r2, c2 = out2["res"], out2["cx"]
    print("  weights: " + "  ".join(f"{a} {b:.1%}" for a, b in r2["fit"]["weights"].items())
          + f"  cash {r2['fit']['w_cash']:.1%}  (IS R^2 {r2['fit']['r2_is']:.3f})")
    print(f"  OOS {r2['oos']['start']} -> {r2['oos']['end']}  n={r2['oos']['n_days']}  "
          f"alpha {r2['oos']['alpha_ann']:+.2%}/yr  (t {r2['oos']['t_hac']:+.2f})")
    print(f"  up-capture {c2['up_capture']:.3f}  down-capture {c2['down_capture']:.3f}  "
          f"smile {c2['smile_bps']:+.1f} bps (t {c2['t_smile']:+.2f})")

    print("\n=== the worst month for the 'bond floor' story ===")
    worst = (fm - rm).sort_values()
    for mo in worst.index[:3]:
        print(f"  {mo}: fund {fm[mo]:+.2%} vs replica {rm[mo]:+.2%}  "
              f"(shortfall {(fm - rm)[mo]:+.2%})")

    print("\n=== synthetic control (machinery proof - never market evidence) ===")
    pl = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=953)[0])
    print(f"  planted (alpha 2%/yr + delta ratchet): OOS alpha {pl['alpha_oos_ann']:+.2%}/yr "
          f"(t {pl['t_oos']:+.2f}), TM gamma {pl['tm_gamma']:+.2f} (t {pl['t_tm_gamma']:+.2f}), "
          f"smile {pl['smile_bps']:+.1f} bps")
    nulls = np.array([
        [d["t_oos"], d["t_tm_gamma"]]
        for d in (st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=953 + s)[0])
                  for s in range(8))
    ])
    print(f"  null x8: alpha t mean {nulls[:, 0].mean():+.2f} (sd {nulls[:, 0].std(ddof=1):.2f}), "
          f"|t|>=2 in {(np.abs(nulls[:, 0]) >= 2).sum()}/8; "
          f"gamma t mean {nulls[:, 1].mean():+.2f}, |t|>=2 in {(np.abs(nulls[:, 1]) >= 2).sum()}/8")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
