"""Offline, deterministic demo — Study 155 (Asset-Turnover).

No network.  Builds a synthetic year × firm panel and shows the study's spine: the
AT top-quintile sort beats a random-portfolio control only when a premium is planted in
the panel, and is indistinguishable from random when no premium exists.  Run:

    python studies/155-asset-turnover/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from asset_turnover import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 155 — Asset-Turnover — synthetic positive control\n")
    print(f"{'Planted premium':>16} | {'Top-Q hedge':>11} {'hit%':>5} {'t':>6} | "
          f"{'rand mean':>9} {'beats rand%':>11}")
    print("-" * 72)

    for premium in (0.00, 0.02, 0.04, 0.06):
        at_sig, fwd, truth = data.synthetic_panel(
            n_firms=200, n_years=14, premium=premium, seed=155
        )
        h = st.quintile_returns(at_sig, fwd, q=0.20)
        s_hedge = st.summarize(h["hedge"])

        rand = st.random_portfolio_returns(at_sig, fwd, q=0.20, n_draws=300, seed=1)
        pct_beats = float((rand < h["hedge"].mean()).mean())

        print(
            f"{premium:>16.2f} | "
            f"{s_hedge['mean']*100:>+10.1f}% "
            f"{s_hedge['hit_rate']*100:>5.0f}% "
            f"{s_hedge['tstat']:>+6.2f} | "
            f"{rand.mean()*100:>+8.1f}% "
            f"{pct_beats*100:>10.0f}%"
        )

    print()
    print("The AT sort is a faithful premium harvester: with a planted edge the top quintile")
    print("beats the random null; with zero premium (null) it is a coin vs the market.")
    print("On the real survivorship-biased tape the signal is NONE -- see docs/results.md.")


if __name__ == "__main__":
    main()
