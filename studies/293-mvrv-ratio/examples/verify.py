"""Real-tape verification -- Study 293 (MVRV-Ratio). Regenerates docs/results.md numbers.

Joins the curated month-end MVRV series to the BTC-USD monthly close, runs the
predictive regression of next-month return on MVRV stretch (with and without a
price-momentum control), reports the average forward return by MVRV band, and
backtests the contrarian "step to cash when over-heated" timing rule against
buy-and-hold.
Network is touched only with --fetch.

    python studies/293-mvrv-ratio/examples/verify.py          # cache-only
    python studies/293-mvrv-ratio/examples/verify.py --fetch  # refresh BTC cache
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mvrv_ratio import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.joined_real(fetch=fetch)
    print(f"Aligned tape: {len(df)} months  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"MVRV fingerprint: {data.fingerprint(data.mvrv_series())}")
    print(f"BTC fingerprint (last row): {data.fingerprint(df)}\n")

    # ---- Predictive regression -------------------------------------------
    print("=== Predictive regression: r(t+1) on MVRV stretch(t) ===")
    r = st.predictive_regression(df)
    print(f"  slope={r['slope_mvrv']:+.4f}  HAC t={r['t_mvrv']:+.2f}  R^2={r['r2']:.4f}  n={r['n']}")

    rc = st.predictive_regression(df, add_price_control=True)
    print("\n=== Horse race vs BTC price momentum ===")
    print(f"  MVRV slope: HAC t={rc['t_mvrv']:+.2f}")
    print(f"  price-mom slope: HAC t={rc['t_price']:+.2f}")

    # ---- Forward return by MVRV band -------------------------------------
    print("\n=== Average next-month BTC return by MVRV band ===")
    tab = st.state_forward_stats(df, high=3.5, low=1.0)
    for band, row in tab.iterrows():
        print(f"  {band:<13}: {row['mean']*100:+.2f}%/mo  hit={row['hit']:.2f}  n={int(row['n'])}")

    # ---- Contrarian timing rule vs buy-and-hold --------------------------
    pos = st.timing_signal(df, high=3.5, low=1.0)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    s_gross = st.summarize(bt["gross"])
    s_net = st.summarize(bt["net"])
    s_bh = st.summarize(bt["bh"])
    print("\n=== Contrarian timing (cash when over-heated) vs buy-and-hold ===")
    print(f"  Time in market: {st.time_in_market(pos):.1%}   avg turnover: {st.turnover(pos):.3f}/mo")
    print(f"  GROSS: {s_gross['mean']*1200:+.1f}%/yr  SR={s_gross['sharpe']*np.sqrt(12):+.2f}  HAC t={s_gross['tstat']:+.2f}")
    print(f"  NET  : {s_net['mean']*1200:+.1f}%/yr  SR={s_net['sharpe']*np.sqrt(12):+.2f}  HAC t={s_net['tstat']:+.2f}")
    print(f"  BHODL: {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*np.sqrt(12):+.2f}  HAC t={s_bh['tstat']:+.2f}")
    excess = bt["net"] - bt["bh"]
    se = st.summarize(excess)
    print(f"  Timing - buy-hold: {se['mean']*1200:+.1f}%/yr  HAC t={se['tstat']:+.2f}")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
