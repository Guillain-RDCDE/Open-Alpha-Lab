"""Real-tape verification — Study 945 (The Hidden Financing). Regenerates docs/results.md.

Reads cached SSO (2x), UPRO (3x), SPY (benchmark), BIL (cash), ^IRX (13-week bill rate)
and ^GSPC (price index, wrong-benchmark cross-check), backs the implied financing rate
out of each wrapper, prints the HAC inference, the block-bootstrap CI, the rolling
rate-cycle pass-through, the era and rate-regime cuts, the expense-ratio sweep, and the
wrapper-versus-margin-account race with its break-even spread. Network only on
``--fetch``.

    python studies/945-leverage-financing-cost/examples/verify.py            # cache-only
    python studies/945-leverage-financing-cost/examples/verify.py --fetch    # refresh
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lev_financing import data, strategy as st  # noqa: E402

BENCH = "SPY"
CASH = "BIL"
RATE = "IRX"


def main(fetch: bool) -> None:
    if fetch:
        data.fetch()
    px = data.load_prices()
    px = px.dropna()  # the common SSO n UPRO n SPY n BIL n IRX n GSPC window
    rets = px[["SSO", "UPRO", "SPY", "BIL", "GSPC"]].pct_change().dropna()
    rate = px[RATE].reindex(rets.index)

    print(f"common window {rets.index[0].date()} -> {rets.index[-1].date()}  "
          f"n={len(rets)}  fp={data.fingerprint(px)}")
    print(f"as-of {data.AS_OF}  |  mean ^IRX {rate.mean():.3f}%  |  "
          f"BIL realised {rets['BIL'].mean() * 252 * 100:.3f}%/yr")

    print("\n=== implied financing (OLS on SPY total return, HAC SEs, index basis) ===")
    est = {}
    for fund, L in data.FUNDS.items():
        e = st.implied_financing(rets[fund], rets[BENCH], L,
                                 expense_ratio=data.EXPENSE_RATIO[fund],
                                 bench_fee=data.SPY_EXPENSE_RATIO, rate_pct=rate)
        est[fund] = e
        print(f"  {fund} ({L}x): beta {e['beta']:.4f} (se {e['beta_se']:.4f}, "
              f"t vs {L} = {e['beta_t_vs_L']:+.2f})  R2 {e['r2']:.5f}  "
              f"resid sd {e['resid_sd_bps']:.1f} bp")
        print(f"        alpha {e['alpha_ann_pct']:+.3f}%/yr (se {e['alpha_se_ann_pct']:.3f}, "
              f"t = {e['alpha_t']:+.2f})  ->  drag {e['drag_ann_pct']:.3f}%/yr")
        print(f"        implied financing {e['implied_financing_pct']:.3f}%  "
              f"= ^IRX {e['mean_rate_pct']:.3f}% {e['spread_over_rate_pct']:+.3f}%  "
              f"(se {e['spread_se_ann_pct']:.3f}, HAC t = {e['spread_t']:+.2f})")
        print(f"        all-in cost per borrowed dollar "
              f"{e['cost_per_borrowed_dollar_pct']:.3f}%  "
              f"(^IRX {e['all_in_spread_over_rate_pct']:+.3f}%, "
              f"HAC t = {e['all_in_spread_t']:+.2f})")
    print("  NOTE: the intercept t above tests only 'the drag is non-zero', which the fee")
    print("        alone guarantees. The claims are the two spread t's — those are the bar.")

    print("\n=== block-bootstrap CI on the financing spread (2000 draws, 21-day blocks) ===")
    for fund, L in data.FUNDS.items():
        ci = st.bootstrap_spread(rets[fund], rets[BENCH], L,
                                 expense_ratio=data.EXPENSE_RATIO[fund],
                                 mean_rate_pct=est[fund]["mean_rate_pct"],
                                 bench_fee=data.SPY_EXPENSE_RATIO, seed=945)
        print(f"  {fund}: spread {ci['point']:+.3f}%  95% CI "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]  share<0 {ci['frac_below_zero']:.3f}")

    print("\n=== wrong-benchmark cross-check (regress on the ^GSPC PRICE index) ===")
    for fund, L in data.FUNDS.items():
        e = st.implied_financing(rets[fund], rets["GSPC"], L,
                                 expense_ratio=data.EXPENSE_RATIO[fund],
                                 bench_fee=0.0, rate_pct=rate)
        print(f"  {fund}: drag {e['drag_ann_pct']:+.3f}%/yr -> implied financing "
              f"{e['implied_financing_pct']:+.3f}% (nonsense: the index's dividend yield "
              f"x {L} is missing from the benchmark)")

    print("\n=== rolling 252-day estimate through the rate cycle ===")
    for fund, L in data.FUNDS.items():
        roll = st.rolling_financing(rets[fund], rets[BENCH], rate, L,
                                    expense_ratio=data.EXPENSE_RATIO[fund],
                                    bench_fee=data.SPY_EXPENSE_RATIO, window=252)
        pt = st.pass_through_regression(roll)
        print(f"  {fund}: spread mean {roll['spread'].mean():+.3f}%  median "
              f"{roll['spread'].median():+.3f}%  sd {roll['spread'].std():.3f}  "
              f"min {roll['spread'].min():+.3f}  max {roll['spread'].max():+.3f}")
        print(f"        pass-through slope {pt['slope']:.3f} (se {pt['slope_se']:.3f})  "
              f"intercept {pt['intercept']:+.3f}%  corr {pt['corr']:.3f}  n {pt['n']}")
        for yr in (2010, 2013, 2016, 2019, 2022, 2024, 2026):
            sl = roll.loc[str(yr)] if str(yr) in roll.index.strftime("%Y") else None
            if sl is not None and len(sl):
                print(f"          {yr}: financing {sl['implied_financing'].mean():5.2f}%  "
                      f"^IRX {sl['rate'].mean():5.2f}%  spread {sl['spread'].mean():+.2f}%  "
                      f"beta {sl['beta'].mean():.3f}")

    print("\n=== era cut (split 2018-01-01) and rate-regime cut (^IRX < 1%) ===")
    for fund, L in data.FUNDS.items():
        eras = st.era_cut(rets[fund], rets[BENCH], rate, L, data.EXPENSE_RATIO[fund],
                          bench_fee=data.SPY_EXPENSE_RATIO, split="2018-01-01")
        regs = st.rate_regime_cut(rets[fund], rets[BENCH], rate, L,
                                  data.EXPENSE_RATIO[fund],
                                  bench_fee=data.SPY_EXPENSE_RATIO, threshold=1.0)
        for tag, e in list(eras.items()) + list(regs.items()):
            if e is None:
                continue
            print(f"  {fund} {tag:6s} n={e['n_days']:5d}  beta {e['beta']:.3f}  "
                  f"financing {e['implied_financing_pct']:6.3f}%  ^IRX "
                  f"{e['mean_rate_pct']:5.3f}%  spread {e['spread_over_rate_pct']:+.3f}%  "
                  f"(spread t {e['spread_t']:+.2f}"
                  f"{'  <-- BELOW THE |t|>=2 BAR' if abs(e['spread_t']) < 2 else ''})")

    print("\n=== expense-ratio sweep (the one non-tape input) ===")
    for fund, L in data.FUNDS.items():
        for row in st.expense_ratio_sweep(rets[fund], rets[BENCH], rate, L,
                                          expense_ratios=(0.75, 0.85,
                                                          data.EXPENSE_RATIO[fund],
                                                          0.95, 1.05),
                                          bench_fee=data.SPY_EXPENSE_RATIO):
            print(f"  {fund} ER {row['expense_ratio_pct']:.2f}% -> financing "
                  f"{row['implied_financing_pct']:.3f}%  spread "
                  f"{row['spread_over_rate_pct']:+.3f}%")

    print("\n=== wrapper vs self-financed margin replication (excess-of-cash both sides) ===")
    for fund, L in data.FUNDS.items():
        for label, m in data.MARGIN_SPREADS.items():
            race = st.replication_race(rets[fund], rets[BENCH], rets[CASH], rate, L,
                                       margin_spread=m, cost_bps=1.0)
            print(f"  {fund} vs margin at ^IRX+{m:.2f}% ({label}): "
                  f"fund exSharpe {race['fund']['sharpe']:+.3f} / DIY "
                  f"{race['replication']['sharpe']:+.3f}  "
                  f"diff {race['diff_ann_pct']:+.2f}%/yr (HAC t {race['t_diff']:+.2f})")
        for c in (0.0, 0.5, 1.0, 2.0, 5.0):
            be = st.breakeven_margin_spread(rets[fund], rets[BENCH], rate, L, cost_bps=c)
            print(f"  {fund} break-even margin spread at {c:.1f} bp one-way: "
                  f"^IRX {be:+.3f}%")
        # The break-even is a FULL-SAMPLE in-sample average. Show how far it moves when
        # the sample is cut, so nobody mistakes it for a stable forward constant.
        cuts = [("2009-2017", rets.index <= pd.Timestamp("2018-01-01")),
                ("2018-2026", rets.index > pd.Timestamp("2018-01-01")),
                ("^IRX<1%", (rate < 1.0).to_numpy()),
                ("^IRX>=1%", (rate >= 1.0).to_numpy())]
        for tag, mask in cuts:
            idx = rets.index[mask]
            be = st.breakeven_margin_spread(rets[fund].loc[idx], rets[BENCH].loc[idx],
                                            rate.loc[idx], L, cost_bps=1.0)
            print(f"        break-even in {tag:9s}: ^IRX {be:+.3f}%")

    print("\n=== synthetic control (machinery proof — never supports the stamp) ===")
    for ss, tag in ((1.0, "planted 75 bp"), (0.0, "null (0 bp)")):
        for L in (2, 3):
            errs, spreads = [], []
            for s in range(6):
                p, t = data.synthetic_panel(signal_strength=ss, seed=945 + s)
                d = st.synthetic_detect(p, t, leverage=L)
                spreads.append(d["spread_over_rate_pct"])
                errs.append(d["spread_error_pct"])
            print(f"  {tag:14s} {L}x: recovered spread mean {np.mean(spreads):+.3f}% "
                  f"(planted {t['planted_spread_pct']:.2f}%)  mean error "
                  f"{np.mean(errs):+.3f}%  max |error| {np.max(np.abs(errs)):.3f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
