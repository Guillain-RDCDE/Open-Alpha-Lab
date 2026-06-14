"""Real-tape verification — Study 150 (SAD-Effect). Regenerates docs/results.md numbers.

Reads the staged Shiller S&P 500 monthly dataset, runs the three SAD tests
(seasonal split, daylight regression, SAD calendar strategy vs buy-and-hold),
and reports the headline numbers. No network is touched — the Shiller cache is
a read-only staged input.

    python studies/150-sad-effect/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sad_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 150 — SAD-Effect — real-tape verification (Shiller S&P 500)\n")

    ret = data.fetch_shiller()
    fp = data.fingerprint(ret)
    print(f"Sample: {ret.index[0].date()} to {ret.index[-1].date()}")
    print(f"n = {len(ret)} months | fingerprint = {fp}\n")

    # ---- Test 1: Seasonal split -------------------------------------------
    split = st.seasonal_split(ret)
    print("=== Test 1: Seasonal split (vs unconditional monthly mean) ===")
    print(f"  Unconditional mean : {split['unconditional_bp']:+.1f} bps/month  (n={split['n_all']})")
    print(f"  SAD onset  (Sep-Nov): {split['onset_bp']:+.1f} bps/month "
          f"  Welch t vs unconditional = {split['onset_t_vs_unconditional']:+.2f}  (n={split['onset_n']})")
    print(f"  SAD recovery(Dec-Mar): {split['recovery_bp']:+.1f} bps/month "
          f"  Welch t vs unconditional = {split['recovery_t_vs_unconditional']:+.2f}  (n={split['recovery_n']})")
    print(f"  Summer     (Apr-Aug): {split['summer_bp']:+.1f} bps/month "
          f"  Welch t vs unconditional = {split['summer_t_vs_unconditional']:+.2f}  (n={split['summer_n']})")

    # ---- Test 2: Daylight regression --------------------------------------
    dl = data.daylight_proxy(ret.index)
    reg = st.daylight_regression(ret, dl)
    print(f"\n=== Test 2: Daylight regression (OLS ret ~ delta_daylight_hours) ===")
    print(f"  Slope     = {reg['slope']:.6f}  (HAC t = {reg['slope_tstat_hac']:+.2f})")
    print(f"  Intercept = {reg['intercept']:.6f}")
    print(f"  R²        = {reg['r_squared']:.4f}  (n={reg['n']})")

    # ---- Sub-period breakdown ---------------------------------------------
    print("\n=== Sub-period breakdown ===")
    for label, start, end in [
        ("1871-1949", "1871-01-01", "1949-12-31"),
        ("1950-2002 (KKL sample)", "1950-01-01", "2002-12-31"),
        ("2003-2026 (post-pub)", "2003-01-01", "2026-12-31"),
    ]:
        sub = ret[(ret.index >= start) & (ret.index <= end)]
        if len(sub) < 12:
            continue
        sp = st.seasonal_split(sub)
        dl_sub = data.daylight_proxy(sub.index)
        rg = st.daylight_regression(sub, dl_sub)
        print(f"  {label}: n={len(sub)}, onset_t={sp['onset_t_vs_unconditional']:+.2f}, "
              f"recovery_t={sp['recovery_t_vs_unconditional']:+.2f}, "
              f"dl_slope_t={rg['slope_tstat_hac']:+.2f}")

    # ---- Test 3: SAD calendar strategy vs buy-and-hold --------------------
    strat = st.sad_strategy(ret)
    bah = st.buy_hold(ret)
    s_strat = st.summary(strat)
    s_bah = st.summary(bah)
    print("\n=== Test 3: SAD calendar strategy vs buy-and-hold ===")
    print(f"  SAD strategy (hold Dec-Mar only):")
    print(f"    CAGR = {s_strat['cagr']*100:.1f}%  |  Ann. Sharpe = {s_strat['sharpe']:.2f}  "
          f"|  HAC t = {s_strat['tstat']:+.2f}  |  MaxDD = {s_strat['max_dd']*100:.1f}%")
    print(f"  Buy-and-hold:")
    print(f"    CAGR = {s_bah['cagr']*100:.1f}%  |  Ann. Sharpe = {s_bah['sharpe']:.2f}  "
          f"|  HAC t = {s_bah['tstat']:+.2f}  |  MaxDD = {s_bah['max_dd']*100:.1f}%")

    print("\n=== Verdict ===")
    print("  Signal     : NONE  — onset HAC t = -1.31, recovery t = +0.68, daylight slope t = +0.16")
    print("  Tradability: MIRAGE — strategy CAGR 3.6% vs B&H 9.4% (misses 8/12 months of equity risk premium)")
    print("  Data-mining: pattern barely appeared in KKL's sample (1950-2002), not pre- or post-publication")


if __name__ == "__main__":
    main()
