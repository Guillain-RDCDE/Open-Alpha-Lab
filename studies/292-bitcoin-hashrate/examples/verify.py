"""Real-tape verification -- Study 292 (Bitcoin-Hashrate). Regenerates docs/results.md numbers.

Joins the curated month-end hash-rate series to the BTC-USD monthly close, runs the
predictive regression of next-month return on hash-rate growth (with and without a
price-momentum control), backtests the "long when hashrate rises" timing rule against
buy-and-hold, and reports turnover and a Hash-Ribbons crossover.
Network is touched only with --fetch.

    python studies/292-bitcoin-hashrate/examples/verify.py          # cache-only
    python studies/292-bitcoin-hashrate/examples/verify.py --fetch  # refresh BTC cache
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_hashrate import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.joined_real(fetch=fetch)
    print(f"Aligned tape: {len(df)} months  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Hashrate fingerprint: {data.fingerprint(data.hashrate_series())}")
    print(f"BTC fingerprint (last row): {data.fingerprint(df)}\n")

    # ---- Predictive regression -------------------------------------------
    print("=== Predictive regression: r(t+1) on hashrate growth(t) ===")
    for lb in (1, 3, 6):
        r = st.predictive_regression(df, lookback=lb)
        print(f"  lookback={lb}mo: slope={r['slope_hash']:+.3f}  HAC t={r['t_hash']:+.2f}  R^2={r['r2']:.3f}  n={r['n']}")

    rc = st.predictive_regression(df, lookback=1, add_price_control=True)
    print("\n=== Horse race vs BTC price momentum ===")
    print(f"  hashrate slope: HAC t={rc['t_hash']:+.2f}")
    print(f"  price-mom slope: HAC t={rc['t_price']:+.2f}")

    # ---- Timing rule vs buy-and-hold -------------------------------------
    pos = st.timing_signal(df, lookback=1)
    bt = st.backtest_timing(df, pos, cost_bps=30.0)
    s_gross = st.summarize(bt["gross"])
    s_net = st.summarize(bt["net"])
    s_bh = st.summarize(bt["bh"])
    print("\n=== Timing (long when hashrate rises) vs buy-and-hold ===")
    print(f"  Time in market: {st.time_in_market(pos):.1%}   avg turnover: {st.turnover(pos):.2f}/mo")
    print(f"  GROSS: {s_gross['mean']*1200:+.1f}%/yr  SR={s_gross['sharpe']*np.sqrt(12):+.2f}  HAC t={s_gross['tstat']:+.2f}")
    print(f"  NET  : {s_net['mean']*1200:+.1f}%/yr  SR={s_net['sharpe']*np.sqrt(12):+.2f}  HAC t={s_net['tstat']:+.2f}")
    print(f"  BHODL: {s_bh['mean']*1200:+.1f}%/yr  SR={s_bh['sharpe']*np.sqrt(12):+.2f}  HAC t={s_bh['tstat']:+.2f}")
    excess = bt["net"] - bt["bh"]
    se = st.summarize(excess)
    print(f"  Timing - buy-hold: {se['mean']*1200:+.1f}%/yr  HAC t={se['tstat']:+.2f}")

    # ---- Hash-Ribbons crossover ------------------------------------------
    pos_r = st.hash_ribbon_signal(df, fast=3, slow=6)
    bt_r = st.backtest_timing(df, pos_r, cost_bps=30.0)
    s_r = st.summarize(bt_r["net"])
    print("\n=== Hash-Ribbons (3/6 MA crossover) ===")
    print(f"  NET: {s_r['mean']*1200:+.1f}%/yr  SR={s_r['sharpe']*np.sqrt(12):+.2f}  long {st.time_in_market(pos_r):.0%} of months")

    print(f"\nFingerprint: {data.fingerprint(df)}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
