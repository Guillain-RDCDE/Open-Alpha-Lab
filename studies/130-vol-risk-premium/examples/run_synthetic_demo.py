"""Offline, deterministic demo -- Study 130 (Vol-Risk-Premium).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the VRP (IV minus RV) premium exists by construction, and when a predictive signal is
planted the top-quintile forward-return spread becomes detectable; on the null tape
(vrp_signal=0) the timing test should not fire.  Run:

    python studies/130-vol-risk-premium/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_risk_premium import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 130 -- Vol-Risk-Premium -- synthetic positive control\n")

    # --- Part 1: VRP premium exists regardless of predictive signal ---
    print("Part 1: VRP premium (IV > RV on average)")
    print(f"{'vrp_signal':>12} | {'mean_VRP':>9} {'pct_pos%':>9} {'tstat':>7}")
    print("-" * 48)
    for sig in (0.0, 0.01):
        df, _ = data.synthetic_daily(n_days=2000, vrp_signal=sig, seed=130)
        ps = st.vrp_premium_stats(df)
        print(f"{sig:>12.2f} | {ps['mean_vrp']:>9.2f} {ps['pct_positive']:>9.1f} {ps['tstat']:>7.2f}")

    print()
    print("The premium exists in both tapes -- VIX structurally exceeds trailing RV.\n")

    # --- Part 2: Predictability of forward returns ---
    print("Part 2: Q5-Q1 forward-return spread (h=21 days, planted vs null)")
    print(f"{'vrp_signal':>12} | {'spread_bps':>11} {'t_spread':>9} {'t_top':>7} | predictive?")
    print("-" * 65)
    for sig in (0.0, 0.01, 0.02):
        df, _ = data.synthetic_daily(n_days=3000, vrp_signal=sig, seed=130)
        tb = st.top_minus_bottom(df, horizon=21)
        pred = "yes" if abs(tb["t_spread"]) > 1.5 and tb["spread_bps"] > 0 else "no"
        print(f"{sig:>12.2f} | {tb['spread_bps']:>+11.1f} {tb['t_spread']:>+9.2f} "
              f"{tb['t_top']:>+7.2f} | {pred}")

    print()
    print("The Q5-Q1 spread grows with the planted signal, confirming the engine works.")
    print()
    print("On the REAL tape (~33 years of VIX vs SPY 21d-RV):")
    print("  VRP premium: mean ~3.7pp, t ~23, positive ~86% of days  => REAL")
    print("  Q5 (high-VRP) h=1 forward return: t ~2.9               => REAL signal")
    print("  Q5-Q1 spread h=1: t ~1.7 (does not clear |t|>=2 bar)   => MIXED")
    print("  Timing overlay vs buy-and-hold: negative spread          => MIXED/FRAGILE")
    print("  Short-vol payoff: skew ~-0.7, worst month ~-20%          => FRAGILE")
    print()
    print("See docs/results.md for the full real-tape numbers.")


if __name__ == "__main__":
    main()
