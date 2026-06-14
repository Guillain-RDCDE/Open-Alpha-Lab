"""Real-tape verification — Study 133 (Crypto-Seasonality). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the BTC-USD daily tape, computes per-month statistics
and HAC t-stats, runs the seasonal long/flat rule vs buy-and-hold, and prints the
numbers that populate docs/results.md. Network is touched only with --fetch.

    python studies/133-crypto-seasonality/examples/verify.py            # cache-only
    python studies/133-crypto-seasonality/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_seasonality import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    bars = data.fetch_daily(fetch=fetch)
    print(f"BTC-USD tape: {bars.index[0].date()} .. {bars.index[-1].date()}  "
          f"n={len(bars)} days  fp={data.fingerprint(bars)}")
    print()

    mret = st.monthly_returns(bars)
    print(f"Monthly obs: {len(mret)}  ({mret['year'].min()}–{mret['year'].max()})")
    print()

    stats = st.month_stats(mret)
    print("=== per-month statistics (vs all-month baseline) ===")
    print(f"{'Mo':4s} {'n':3s} {'mean%':7s} {'median%':8s} {'std%':6s} {'HAC t':7s} {'naive':5s} {'bonf':4s}")
    for _, row in stats.iterrows():
        print(f"{row['month_name']:3s}  {row['n']:3d}  {row['mean_pct']:+7.1f}%  "
              f"{row['median_pct']:+7.1f}%  {row['std_log']*100:5.1f}%  "
              f"{row['tstat_vs_baseline']:+6.2f}   "
              f"{'*' if row['naive_sig'] else ' ':1s}     {'*' if row['bonferroni_sig'] else ' ':1s}")

    n_naive = stats["naive_sig"].sum()
    n_bonf = stats["bonferroni_sig"].sum()
    print(f"\nNaive sig (|t|>=2): {n_naive}/12  |  Bonferroni sig (|t|>=3.1): {n_bonf}/12")
    print("(Bonferroni threshold for 12 tests at alpha=0.05: |t| ~ 3.1)")

    bh_s = st.summarize(mret["log_ret"].to_numpy(), label="buy_and_hold")
    print(f"\nAll-month B&H: mean={bh_s['mean_pct']:+.2f}%/mo  "
          f"ann_sharpe={bh_s['sharpe_ann']:+.2f}  t={bh_s['tstat']:+.2f}  n={bh_s['n_months']}")

    print("\n=== seasonal long/flat vs buy-and-hold ===")
    for long_months, label in [
        ([10], "Uptober-only (Oct)"),
        ([7, 10], "Jul+Oct"),
        ([10, 4, 2, 7, 11], "Best-5 in-sample"),
        ([9], "Sep-only (long Sep = losing month)"),
    ]:
        result = st.bh_vs_strategy(mret, long_months=long_months)
        s = result["strategy"]
        print(f"  {label:30s}: sharpe={s['sharpe_ann']:+.2f}  diff_t={result['diff_tstat']:+.2f}")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
