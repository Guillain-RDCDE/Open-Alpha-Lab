"""Offline, deterministic demo — Study 208 (Gold-Miners).

No network. Builds a synthetic GLD/GDX tape and shows the study's three-pronged spine:
1. Beta/alpha estimation — does OLS recover the planted leverage and operational drag?
2. Asymmetric beta — does the engine detect more downside than upside beta when planted?
3. Timing rule — does the GDX/GLD ratio momentum rule beat a simple GLD hold?

Run:
    python studies/208-gold-miners/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gold_miners import data, strategy as st  # noqa: E402


def _bar(label: str, width: int = 60) -> None:
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print("=" * width)


def main() -> None:
    print("Study 208 — Gold-Miners — synthetic positive control\n")
    print("Planted parameters -> estimated values (on deterministic offline tapes)\n")

    # ---- Test 1: Beta / Alpha recovery ----
    _bar("Test 1: OLS beta / alpha — 'leveraged gold' claim")
    print(f"{'planted beta':>15} | {'planted alpha':>14} | {'est. beta':>9} {'t(beta)':>8} | "
          f"{'est. alpha %/yr':>16} {'t(alpha)':>9}")
    print("-" * 80)
    for lev, alph in [(0.0, 0.0), (1.0, 0.0), (1.5, -0.04), (2.0, -0.08)]:
        prices, _ = data.synthetic_daily(n_years=20, leverage=lev, alpha_ann=alph,
                                          asymmetry=0.0, seed=208)
        r = st.beta_alpha(prices)
        print(f"{lev:>15.1f} | {alph*100:>13.1f}% | {r['beta']:>+9.4f} {r['t_beta']:>+8.2f} | "
              f"{r['alpha_ann_pct']:>+15.2f}% {r['t_alpha']:>+9.2f}")

    print("\nThe engine recovers planted beta and negative alpha reliably (|t| > 2).")

    # ---- Test 2: Asymmetric beta ----
    _bar("Test 2: Leverage asymmetry — 'more downside than upside'")
    print(f"{'asymmetry':>12} | {'beta_up':>8} {'beta_dn':>8} {'diff':>8} {'t(diff)':>8} | claim supported?")
    print("-" * 72)
    for asym in (0.0, 0.3, 0.6, -0.3):
        prices, _ = data.synthetic_daily(n_years=20, leverage=1.5, asymmetry=asym,
                                          alpha_ann=0.0, seed=208)
        r = st.asymmetric_beta(prices)
        supported = "yes" if r["t_asymmetry"] > 2.0 else "no"
        print(f"{asym:>12.1f} | {r['beta_up']:>+8.4f} {r['beta_dn']:>+8.4f} "
              f"{r['beta_diff']:>+8.4f} {r['t_asymmetry']:>+8.2f} | {supported}")

    print("\nEngine detects planted asymmetry (t > 2 when asymmetry >= 0.3 with n=20yr).")

    # ---- Test 3: Timing rule ----
    _bar("Test 3: GDX/GLD timing rule vs buy-and-hold GLD")
    print(f"{'leverage':>10} | {'GLD Sh':>7} {'GDX Sh':>7} {'Timing Sh':>10} | timing beats GLD?")
    print("-" * 65)
    for lev in (0.0, 1.0, 1.5, 2.0):
        prices, _ = data.synthetic_daily(n_years=20, leverage=lev, alpha_ann=-0.04,
                                          asymmetry=0.2, seed=208)
        res = st.compare_strategies(prices, sma_n=200, cost_bps=5.0)
        beats = "yes" if res["timing"]["sharpe"] > res["gld"]["sharpe"] else "no"
        print(f"{lev:>10.1f} | {res['gld']['sharpe']:>+7.3f} {res['gdx']['sharpe']:>+7.3f} "
              f"{res['timing']['sharpe']:>+10.3f} | {beats}")

    print("\nWith negative alpha and high idio-vol, the timing rule rarely beats GLD.")
    print("On the real tape (see docs/results.md) it underperforms GLD (Sharpe 0.13 vs 0.49).")


if __name__ == "__main__":
    main()
