"""Offline, deterministic demo — Study 199 (Sales-Growth).

No network. Builds a synthetic firm-year panel and shows the study's spine:
the low-growth (value) portfolio only outperforms the high-growth (glamour)
portfolio when a real LSV premium is planted in the tape. On a null panel
(premium = 0) the quintile sort is statistically indistinguishable from a
random draw across firms.

    python studies/199-sales-growth/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sales_growth import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 199 — Sales-Growth — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'lo-hi hedge':>11} {'t':>6} | "
        f"{'random excess':>13} {'beats coin?':>12}"
    )
    print("-" * 72)
    for prem in (0.0, -0.04, -0.08, -0.12, 0.04):
        sig, fwd, truth = data.synthetic_panel(
            n_firms=300, n_years=25, premium=prem, seed=199
        )
        h = st.quintile_returns(sig, fwd, q=0.20)
        s = st.summary(h["hedge"])
        rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=200, seed=1)
        rand_exc = float(rand.mean())
        beats = "yes" if s["mean"] > rand_exc + 0.01 and s["tstat"] > 1.5 else "no"
        print(
            f"{prem:>18.2f} | {s['mean']*100:>+10.1f}% {s['tstat']:>+6.2f} | "
            f"{rand_exc*100:>+12.1f}%  {beats:>12}"
        )

    print()
    print("The engine detects the effect when it exists (-0.08, -0.12) and prints")
    print("near-zero when there is none (0.0). On the real EDGAR tape (~326 tickers,")
    print("2007-2025) the hedge is about -0.4%/yr -- see docs/results.md.")
    print("NOTE: real results are SURVIVORSHIP-BIASED (upper bound only).")
    print("NOTE: high-growth stocks do NOT underperform on this panel (opposite of LSV).")


if __name__ == "__main__":
    main()
