"""Offline, deterministic demo — Study 121 (Magic-Formula).

No network. Builds a synthetic firm panel and shows the study's spine:
the Magic-Formula top decile only beats a random draw of the same size
when a real quality+cheapness premium is planted in the tape. On a null
panel (premium = 0) the top decile is statistically indistinguishable from
a coin flip across firms.

    python studies/121-magic-formula/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from magic_formula import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 121 — Magic-Formula — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'top excess':>10} {'t':>6} | "
        f"{'random excess':>13} {'beats coin?':>12}"
    )
    print("-" * 70)
    for prem in (0.0, 0.03, 0.06, 0.10, -0.04):
        sig, fwd, truth = data.synthetic_panel(
            n_firms=300, n_years=25, premium=prem, seed=121
        )
        h = st.quantile_hedge(sig, fwd, q=0.10)
        s = st.summary(h["hedge"])
        rand = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=200, seed=1)
        rand_exc = float(rand.mean())
        beats = "yes" if s["mean"] > rand_exc + 0.01 and s["tstat"] > 1.5 else "no"
        print(
            f"{prem:>18.2f} | {s['mean']*100:>+9.1f}% {s['tstat']:>+6.2f} | "
            f"{rand_exc*100:>+12.1f}%  {beats:>12}"
        )

    print()
    print("The engine detects the premium when it exists (+0.06, +0.10) and prints")
    print("near-zero when there is none (0.0). On the real EDGAR tape (S&P 500, ~17yr)")
    print("the top-decile excess is only +1.9%/yr, HAC t=0.67 — see docs/results.md.")


if __name__ == "__main__":
    main()
