"""Real-tape verification — Study 209 (ETH-BTC-Ratio). Regenerates docs/results.md numbers.

Fetches (or reads from cache) ETH-USD and BTC-USD daily tapes, runs the 20-day ratio
momentum rotation vs the 50/50 daily-rebalanced baseline, sweeps lookbacks and costs, and
computes the alpha t-stat vs the 50/50 baseline. Network is touched only with --fetch.

    python studies/209-eth-btc-ratio/examples/verify.py            # cache-only
    python studies/209-eth-btc-ratio/examples/verify.py --fetch    # refresh the tapes
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from eth_btc_ratio import data, strategy as st  # noqa: E402
from quantlab.stats import sharpe_ci_bootstrap  # noqa: E402


def main(fetch: bool) -> None:
    bars = data.load_pair(fetch=fetch)

    if fetch:
        print(f"BTC: {bars.index[0].date()}..{bars.index[-1].date()} fp={data.fingerprint(bars, 'BTC-USD')}")
        print(f"ETH: {bars.index[0].date()}..{bars.index[-1].date()} fp={data.fingerprint(bars, 'ETH-USD')}")
        print()

    bt = st.backtest_rotation(bars, lookback=20, cost_bps=5.0)
    bt0 = st.backtest_rotation(bars, lookback=20, cost_bps=0.0)
    s_rot = st.summarize(bt["port_net"])
    s_gross = st.summarize(bt0["port_gross"])
    s_50  = st.summarize(bt["fifty_fifty"])
    s_btc = st.summarize(bt["bnh_btc"])
    s_eth = st.summarize(bt["bnh_eth"])
    alpha = st.alpha_vs_baseline(bt["port_net"], bt["fifty_fifty"])
    ci = sharpe_ci_bootstrap(bt["port_net"], periods_per_year=365, seed=209)

    print(f"=== rotation (20d, 5bps) | n={s_rot['n']} days ===")
    print(f"  Sharpe={s_rot['sharpe_ann']:.3f}  t={s_rot['tstat']:+.2f}  ann={s_rot['mean_ann_pct']:.1f}%  maxDD={s_rot['max_dd_pct']:.1f}%")
    print(f"  Bootstrap 95% CI Sharpe: [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  frac_neg={ci['frac_negative']:.3f}")
    print()
    print(f"  Gross (0bps): Sharpe={s_gross['sharpe_ann']:.3f}  t={s_gross['tstat']:+.2f}  ann={s_gross['mean_ann_pct']:.1f}%")
    print()
    print(f"=== benchmarks ===")
    print(f"  50/50:        Sharpe={s_50['sharpe_ann']:.3f}  t={s_50['tstat']:+.2f}  ann={s_50['mean_ann_pct']:.1f}%  maxDD={s_50['max_dd_pct']:.1f}%")
    print(f"  BTC buy-hold: Sharpe={s_btc['sharpe_ann']:.3f}  t={s_btc['tstat']:+.2f}  ann={s_btc['mean_ann_pct']:.1f}%  maxDD={s_btc['max_dd_pct']:.1f}%")
    print(f"  ETH buy-hold: Sharpe={s_eth['sharpe_ann']:.3f}  t={s_eth['tstat']:+.2f}  ann={s_eth['mean_ann_pct']:.1f}%  maxDD={s_eth['max_dd_pct']:.1f}%")
    print()
    print(f"=== alpha vs 50/50 ===")
    print(f"  alpha={alpha['alpha_daily_bps']:.2f}bps/day  alpha_ann={alpha['alpha_ann_pct']:.1f}%  beta={alpha['beta']:.3f}  R2={alpha['r_squared']:.3f}  t_alpha={alpha['t_alpha']:+.2f}")
    print()
    print("=== lookback sweep (5bps cost) ===")
    sweep = st.lookback_sweep(bars, lookbacks=[5, 10, 20, 30, 60, 90, 120], cost_bps=5.0)
    print(sweep.to_string(index=False))
    print()
    print("=== cost sweep (20d lookback) ===")
    for c in [0, 5, 25, 50, 100]:
        btc_ = st.backtest_rotation(bars, lookback=20, cost_bps=c)
        s = st.summarize(btc_["port_net"])
        print(f"  cost={c:3d}bps  Sharpe={s['sharpe_ann']:.3f}  t={s['tstat']:+.2f}  ann={s['mean_ann_pct']:.1f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
