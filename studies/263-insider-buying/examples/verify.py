"""Real-tape verification -- Study 263 (Insider-Buying). Regenerates docs/results.md numbers.

Pairs the curated monthly aggregate insider buy/sell ratio with S&P 500 (^GSPC)
price-only monthly returns, runs the predictive HAC regression, the tercile sort, the
long/flat timing overlay vs buy-and-hold, the sub-period breakdown, and the synthetic
positive control. The buy/sell ratio is a CURATED PROXY (see insider_buying/data.py),
so the Signal axis is capped at WEAK by construction.

Network is touched only with --fetch.

    python studies/263-insider-buying/examples/verify.py          # cache-only
    python studies/263-insider-buying/examples/verify.py --fetch  # refresh ^GSPC
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insider_buying import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    g = data.fetch_sp500(fetch=fetch)
    df = data.build_real_panel(g)
    print(f"Panel: {len(df)} months  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Ratio fingerprint: {data.fingerprint(df['bs_ratio'])}")
    print("(Curated buy/sell ratio -- proxy, not a live Form-4 feed -> Signal capped at WEAK)\n")

    # ---- Predictive regression --------------------------------------------
    reg = st.predictive_regression(df)
    print("=== Predictive regression: next-month return ~ standardised ratio ===")
    print(f"slope = {reg['beta']:+.4f}/mo (per 1sigma)  HAC t = {reg['tstat']:+.2f}  "
          f"R^2 = {reg['r2']:.4f}  n = {reg['n']}")

    # ---- Contemporaneous vs predictive correlation ------------------------
    print("\n=== Contemporaneous vs predictive correlation ===")
    print(f"ratio vs same-month return : {df['bs_ratio'].corr(df['ret']):+.3f}")
    print(f"ratio vs next-month return : {df['bs_ratio'].corr(df['fwd_ret']):+.3f}")

    # ---- Tercile sort ------------------------------------------------------
    print("\n=== Tercile sort (next-month return by ratio bucket) ===")
    terc = st.tercile_returns(df)
    for b in terc.index:
        row = terc.loc[b]
        print(f"  {b:>14}: {row['mean_ann']*100:+6.1f}%/yr  t={row['tstat']:+.2f}  n={int(row['n'])}")

    # ---- Sub-periods -------------------------------------------------------
    print("\n=== Sub-period predictive regression ===")
    for lab, a, b in [("2003-2010", "2003", "2010"),
                      ("2011-2017", "2011", "2017"),
                      ("2018-2025", "2018", "2025")]:
        r = st.predictive_regression(df.loc[a:b])
        print(f"  {lab}: slope={r['beta']:+.4f}/mo  HAC t={r['tstat']:+.2f}  n={r['n']}")

    # ---- Timing overlay vs buy-and-hold -----------------------------------
    print("\n=== Long/flat timing overlay vs buy-and-hold ===")
    for bps in (5.0, 10.0):
        ov = st.timing_overlay(df, lookback=24, one_way_bps=bps)
        sn = st.summarize(ov["net"])
        print(f"  overlay net @{bps:g}bps: {sn['mean_ann']*100:+.2f}%/yr  "
              f"SR(ann)={sn['sharpe_ann']:+.2f}  HAC t={sn['tstat']:+.2f}  maxDD={sn['max_drawdown']*100:.0f}%")
    ov = st.timing_overlay(df, lookback=24, one_way_bps=5.0)
    sb = st.summarize(ov["bah"])
    print(f"  buy-and-hold       : {sb['mean_ann']*100:+.2f}%/yr  SR(ann)={sb['sharpe_ann']:+.2f}  "
          f"maxDD={sb['max_drawdown']*100:.0f}%")
    print(f"  overlay turnover   : {st.overlay_turnover(ov):.1%} of months switch")
    print(f"  fraction long      : {ov['pos'].mean():.1%}")

    # ---- Positive control --------------------------------------------------
    print("\n=== Positive control (synthetic) ===")
    dfp, _ = data.synthetic_panel(beta=0.06, seed=263)
    dfn, _ = data.synthetic_panel(beta=0.0, seed=263)
    print(f"  planted beta=+0.06: HAC t = {st.predictive_regression(dfp)['tstat']:+.2f}")
    print(f"  null     beta= 0.0: HAC t = {st.predictive_regression(dfn)['tstat']:+.2f}")

    print("\nVerdict: Signal NONE (HAC t = 0.49, wrong-sign tercile, curated proxy); "
          "Tradability MIRAGE (overlay loses to buy-and-hold).")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
