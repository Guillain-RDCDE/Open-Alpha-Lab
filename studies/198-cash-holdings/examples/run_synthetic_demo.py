"""Offline, deterministic demo — Study 198 (Cash-Holdings).

No network. Builds a synthetic firm panel and shows the study's spine:
the high-cash (high-cash-to-assets) portfolio only outperforms the
low-cash portfolio when a real effect is planted in the tape. On a null
panel (premium = 0) the quintile sort is statistically indistinguishable
from a coin flip across firms.

    python studies/198-cash-holdings/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_holdings import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 198 — Cash-Holdings — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'hi-lo hedge':>11} {'t':>6} | "
        f"{'random excess':>13} {'beats coin?':>12}"
    )
    print("-" * 72)
    for prem in (0.0, 0.04, 0.08, 0.12, -0.04):
        sig, fwd, truth = data.synthetic_panel(
            n_firms=300, n_years=25, premium=prem, seed=198
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
    print("The engine detects the effect when it exists (+0.08, +0.12) and prints")
    print("near-zero when there is none (0.0). On the real EDGAR tape (S&P 500, ~18yr)")
    print("the high-minus-low-cash hedge is +12.4%/yr -- see docs/results.md.")
    print("NOTE: real results are SURVIVORSHIP-BIASED (upper bound only) and partly")
    print("      driven by a latent tech-sector factor, not a pure cash premium.")


if __name__ == "__main__":
    main()
