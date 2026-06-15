"""Offline, deterministic demo — Study 204 (Talmud-Portfolio).

No network. Builds a synthetic three-asset tape and shows the study's spine in one
screen: the Talmud 1/3–1/3–1/3 blend reduces max drawdown when regime diversification
is planted, and how adding REITs compares to a simpler 60/40 allocation. Run:

    python studies/204-talmud-portfolio/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from talmud_portfolio import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 204 — Talmud-Portfolio — synthetic positive control\n")
    print(f"{'cycle_strength':>15} | {'portfolio':>20} | {'CAGR':>6} {'Sharpe':>7} {'MaxDD':>7}")
    print("-" * 65)

    for cycle in (0.0, 0.5, 1.0):
        frame, truth = data.synthetic_three_asset(n_years=15, cycle_strength=cycle, seed=204)
        ret = st.to_returns(frame).rename(columns={"REIT": "VNQ", "STK": "SPY", "BOND": "BND"})
        rf = ret["BND"]

        portfolios = {
            "Talmud 1/3-1/3-1/3": st.talmud_portfolio(ret, cost_bps=10.0),
            "60/40 SPY/BND":       st.sixty_forty(ret, cost_bps=10.0),
            "SPY 100%":            st.spy_only(ret),
        }
        for name, port in portfolios.items():
            s = st.portfolio_stats(port, rf=rf)
            print(f"{cycle:>15.1f} | {name:>20} | "
                  f"{s['cagr']*100:>5.1f}% {s['sharpe']:>+7.2f} {s['max_dd']*100:>6.1f}%")
        print()

    print("Positive control: with planted regime cycles the Talmud blend's diversification")
    print("benefit (lower drawdown, smoother equity curve) becomes more pronounced.")
    print("On the real tape VNQ amplifies equity crashes rather than hedging them,")
    print("making the Talmud blend WORSE than 60/40 on risk-adjusted returns.")
    print("See docs/results.md for the full real-tape teardown.")


if __name__ == "__main__":
    main()
