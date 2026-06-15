"""Offline, deterministic demo — Study 207 (REITs-Diversifier).

No network. Builds a synthetic three-asset world (equity / REIT / bond) and shows
the study's spine in one screen: a REIT sleeve improves the 60/40 benchmark ONLY
when genuine independent diversification is planted — when REITs are pure equity-
beta (diversification=0) the sleeve adds nothing, and on the real tape REITs behave
far more like equity-beta than genuine diversifiers. Run:

    python studies/207-reits-diversifier/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reits_diversifier import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 207 — REITs-Diversifier — synthetic positive control\n")
    print(
        f"{'diversification':>18} | {'sleeve CAGR':>11} {'bench CAGR':>10} | "
        f"{'sleeve Sharpe':>13} {'bench Sharpe':>12} | "
        f"{'dd improve':>10} {'HAC t':>8}"
    )
    print("-" * 100)

    for divers in (0.0, 0.5, 1.0, 2.0):
        prices, _ = data.synthetic_portfolio(n_years=20, diversification=divers, seed=207)
        res = st.summarize_portfolio(
            prices,
            weights_sleeve={"EQ": 0.60, "REIT": 0.20, "BOND": 0.20},
            weights_bench={"EQ": 0.60, "BOND": 0.40},
            cost_bps=1.0,
        )
        dd_improve = res["dd_diff"]  # negative = sleeve has less drawdown = better
        print(
            f"{divers:>18.1f} | {res['sleeve_cagr']:>+11.1%} {res['bench_cagr']:>+10.1%} | "
            f"{res['sleeve_sharpe']:>+13.3f} {res['bench_sharpe']:>+12.3f} | "
            f"{dd_improve:>+10.3f} {res['hac_t']:>+8.2f}"
        )

    print()
    print("With diversification=0 the REIT sleeve is pure equity-beta: no benefit over 60/40.")
    print("With diversification>0 genuine independent income lifts the sleeve above the benchmark.")
    print("On the real tape VNQ's correlation to SPY is ~0.75 — closer to equity-beta than diversifier.")
    print("See docs/results.md for the real-tape numbers and verdict.")


if __name__ == "__main__":
    main()
