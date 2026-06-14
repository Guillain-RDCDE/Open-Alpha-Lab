"""Real-tape verification -- Study 130 (Vol-Risk-Premium). Regenerates docs/results.md numbers.

Fetches (or reads from cache) the daily VIX and SPY tape, computes the VRP premium
statistics, quintile-sorted forward returns, timing overlay, and the short-vol payoff
distribution.  Network is touched only with --fetch.

    python studies/130-vol-risk-premium/examples/verify.py            # cache-only
    python studies/130-vol-risk-premium/examples/verify.py --fetch    # refresh the tape
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_risk_premium import data, strategy as st  # noqa: E402


def main(fetch: bool) -> None:
    df = data.fetch_daily(fetch=fetch, cache_dir=os.path.join(
        os.path.dirname(__file__), "..", "_cache"
    ))
    if fetch:
        print(f"VIX/SPY tape: {df.index[0].date()}..{df.index[-1].date()} "
              f"fp={data.fingerprint(df)} n={len(df)}")
        print()

    print("=== VRP Premium (IV - trailing 21d RV) ===")
    ps = st.vrp_premium_stats(df)
    print(f"  n={ps['n']}  mean_VRP={ps['mean_vrp']:+.3f}pp  std={ps['std_vrp']:.3f}pp  "
          f"pct_pos={ps['pct_positive']:.1f}%  HAC t={ps['tstat']:.2f}")
    print(f"  mean_VIX={ps['mean_vix']:.2f}pp  mean_RV={ps['mean_rv']:.2f}pp")

    print("\n=== Q5 minus Q1 VRP-quintile spread (no look-ahead) ===")
    for h in [1, 5, 21]:
        tb = st.top_minus_bottom(df, horizon=h)
        print(f"  h={h:2d}d  spread={tb['spread_bps']:+.1f}bps  t={tb['t_spread']:+.2f}  "
              f"t_top={tb['t_top']:+.2f}  t_bot={tb['t_bot']:+.2f}")

    print("\n=== Timing overlay (long SPY when VRP > rolling median) ===")
    to0 = st.timing_overlay(df, cost_bps=0)
    to1 = st.timing_overlay(df, cost_bps=1)
    passive = st.summarize(to0["r_passive"].dropna(), "passive")
    active0 = st.summarize(to0["r_active"].dropna(), "active_0cost")
    active1 = st.summarize(to1["r_active"].dropna(), "active_1bps")
    spread0 = st.summarize(to0["r_spread"].dropna(), "spread")
    print(f"  passive (buy-and-hold):  Sharpe {passive['sharpe_ann']:.2f}  "
          f"mean {passive['mean_bps']:+.1f}bps  t {passive['t_stat']:+.2f}")
    print(f"  active (0 cost):         Sharpe {active0['sharpe_ann']:.2f}  "
          f"mean {active0['mean_bps']:+.1f}bps  t {active0['t_stat']:+.2f}")
    print(f"  active (1bp cost):       Sharpe {active1['sharpe_ann']:.2f}  "
          f"mean {active1['mean_bps']:+.1f}bps  t {active1['t_stat']:+.2f}")
    print(f"  spread active-passive:   mean {spread0['mean_bps']:+.1f}bps  t {spread0['t_stat']:+.2f}  "
          f"n_switches={int(to0['switched'].sum())}")

    print("\n=== Short-vol payoff distribution (21-day non-overlapping periods) ===")
    sv = st.short_vol_payoff(df, horizon=21)
    print(f"  n={sv['n']}  mean={sv['mean_ret_pct']:.2f}%  skew={sv['skew']:.2f}  "
          f"kurt={sv['kurt']:.2f}")
    print(f"  pct_negative={sv['pct_negative']:.1f}%  p05={sv['p05']:.1f}%  "
          f"worst_period={sv['worst']:.1f}%")


if __name__ == "__main__":
    main(fetch="--fetch" in sys.argv)
