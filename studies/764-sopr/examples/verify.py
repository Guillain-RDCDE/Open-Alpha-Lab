"""Real-tape verification -- Study 764 (SOPR). Regenerates docs/results.md numbers.

Joins the curated month-end SOPR series to the BTC-USD monthly close, runs the
predictive regression of next-month return on SOPR stretch (with and without a
price-momentum control), reports the average forward return by SOPR band,
backtests the ">1 / <1" regime timing rule against buy-and-hold, and runs the
time-shuffle placebo.
Network is touched only with --fetch.

    python studies/764-sopr/examples/verify.py          # cache-only
    python studies/764-sopr/examples/verify.py --fetch  # refresh BTC cache
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sopr import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.joined_real(fetch=fetch)
    print(f"Aligned tape: {len(df)} months  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"SOPR fingerprint: {data.fingerprint(data.sopr_series())}")
    print(f"BTC fingerprint (last row): {data.fingerprint(df)}\n")

    # ---- Predictive regression -------------------------------------------
    print("=== Predictive regression: r(t+1) on SOPR stretch(t) ===")
    r = st.predictive_regression(df)
    print(f"  slope={r['slope_sopr']:+.4f}  HAC t={r['t_sopr']:+.2f}  R^2={r['r2']:.4f}  n={r['n']}")

    rc = st.predictive_regression(df, add_price_control=True)
    print("\n=== Horse race vs BTC price momentum ===")
    print(f"  SOPR slope: HAC t={rc['t_sopr']:+.2f}")
    print(f"  price-mom slope: HAC t={rc['t_price']:+.2f}")

    # ---- Forward return by SOPR band -------------------------------------
    print("\n=== Average next-month BTC return by SOPR band ===")
    tab = st.state_forward_stats(df, high=1.02, low=0.98)
    for band, row in tab.iterrows():
        print(f"  {band:<13}: {row['mean']*100:+.2f}%/mo  hit={row['hit']:.2f}  n={int(row['n'])}")

    # ---- Regime timing rule vs buy-and-hold ------------------------------
    pos = st.timing_signal(df, thresh=1.0)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    s_gross = st.summarize(bt["gross"])
    s_net = st.summarize(bt["net"])
    s_bh = st.summarize(bt["bh"])
    print("\n=== Regime timing (long when SOPR>=1) vs buy-and-hold ===")
    print(f"  Time in market: {st.time_in_market(pos):.1%}   avg turnover: {st.turnover(pos):.3f}/mo")
    print(f"  GROSS: {s_gross['mean']*1200:+.1f}%/yr  SR={s_gross['sharpe']*np.sqrt(12):+.2f}  HAC t={s_gross['tstat']:+.2f}")
    print(f"  NET  : {s_net['mean']*1200:+.1f}%/yr  SR={s_net['sharpe']*np.sqrt(12):+.2f}  HAC t={s_net['tstat']:+.2f}")
    print(f"  BHODL: {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*np.sqrt(12):+.2f}  HAC t={s_bh['tstat']:+.2f}")
    excess = bt["net"] - bt["bh"]
    se = st.summarize(excess)
    print(f"  Timing - buy-hold: {se['mean']*1200:+.1f}%/yr  HAC t={se['tstat']:+.2f}")

    # ---- Threshold sensitivity -------------------------------------------
    print("\n=== Threshold sensitivity (net %/yr) ===")
    for th in (0.99, 1.00, 1.01):
        p = st.timing_signal(df, thresh=th)
        b = st.backtest_timing(df, p, cost_bps=30.0)
        s = st.summarize(b["net"])
        print(f"  thresh={th:.2f}: net {s['mean']*1200:+.1f}%/yr  SR={s['sharpe']*np.sqrt(12):+.2f}  long {st.time_in_market(p):.0%}")

    # ---- Placebo: shuffle SOPR in time -----------------------------------
    pl = st.placebo_edge(df, n_shuffles=2000)
    print("\n=== Placebo: shuffle SOPR in time (2000 draws) ===")
    print(f"  real edge   : {pl['real_edge_mo']*100:+.2f}%/mo")
    print(f"  placebo mean: {pl['placebo_mean_mo']*100:+.2f}%/mo  std {pl['placebo_std_mo']*100:.2f}pp")
    print(f"  two-sided empirical p: {pl['p_value']:.3f}")

    # ---- Positive & null control -----------------------------------------
    ds, _ = data.synthetic_series(beta=2.0, seed=764)
    rs = st.predictive_regression(ds)
    dn, _ = data.synthetic_series(beta=0.0, seed=764)
    rn = st.predictive_regression(dn)
    print("\n=== Synthetic controls (engine truthfulness) ===")
    print(f"  planted momentum beta=2.0: slope={rs['slope_sopr']:+.3f}  HAC t={rs['t_sopr']:+.2f}")
    print(f"  null beta=0.0            : slope={rn['slope_sopr']:+.3f}  HAC t={rn['t_sopr']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
