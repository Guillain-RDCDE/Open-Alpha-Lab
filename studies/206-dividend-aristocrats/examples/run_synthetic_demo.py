"""Offline, deterministic demo — Study 206 (Dividend-Aristocrats).

No network.  Builds two synthetic price series (aristocrat and benchmark) and shows
the study's spine in one screen: the excess-return t-stat and alpha decomposition
are near zero when no alpha is planted, and clearly positive when it is.  Run:

    python studies/206-dividend-aristocrats/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_aristocrats import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 206 — Dividend-Aristocrats — synthetic positive control\n")
    print(
        f"{'planted alpha':>16} | {'excess CAGR%':>13} {'t-stat':>8} "
        f"{'OLS alpha bps':>14} {'t_alpha':>9} | detects?"
    )
    print("-" * 80)

    for alpha_bps in (0.0, 1.0, 2.0, 3.0, -1.0):
        prices, truth = data.synthetic_daily(n_years=12, alpha_bps=alpha_bps, seed=206)
        diff = st.excess_return(prices)
        s = st.summarize(diff, label=f"{alpha_bps:+.1f} bps/day")
        ols = st.beta_alpha_ols(prices)

        excess_cagr = s["cagr_pct"]   # annualised excess return
        t = s["tstat"]
        ols_alpha = ols["alpha_daily_bps"]
        t_alpha = ols["t_alpha"]
        detects = "YES" if t > 2.0 else ("yes?" if t > 1.5 else "no")

        print(
            f"{alpha_bps:>+16.1f} | {excess_cagr:>+13.2f}% {t:>+8.2f} "
            f"{ols_alpha:>+14.2f} {t_alpha:>+9.2f} | {detects}"
        )

    print()
    print("Spine: t-stat is near zero when no alpha is planted (the null); it grows")
    print("monotonically with planted alpha and clears |t|=2 around +2-3 bps/day.")
    print("On the real NOBL/SPY tape the planted alpha is effectively zero.")
    print("See docs/results.md for the real-tape verdict.")


if __name__ == "__main__":
    main()
