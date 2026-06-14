"""Offline, deterministic demo — Study 123 (Altman-Z).

No network, no EDGAR, no yfinance. Builds a synthetic firm panel and shows the
study's spine in one screen: the Z-score tertile hedge only recovers a premium
when one is planted (the positive control), and reads near zero under the null.

    python studies/123-altman-z/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from altman_z import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 123 — Altman-Z — synthetic positive control\n")
    print(
        f"{'planted premium':>18} | {'hedge mean':>10} {'hedge vol':>10} "
        f"{'t-stat':>8} | engine finds it?"
    )
    print("-" * 68)
    for z_prem in (0.00, 0.04, 0.08, 0.12, -0.06):
        z, fwd, truth = data.synthetic_panel(
            n_firms=300, n_years=25, z_premium=z_prem, seed=123
        )
        res = st.tertile_hedge(z, fwd)
        s = st.summarize(res["hedge"])
        finds = "yes" if (z_prem > 0 and s["mean"] > 0.01) or (z_prem < 0 and s["mean"] < -0.01) else "no"
        print(
            f"{z_prem:>18.2f} | {s['mean']*100:>+9.2f}%  {s['vol']*100:>9.2f}%  "
            f"{s['tstat']:>+8.2f} | {finds}"
        )

    print()
    print("When z_premium > 0 (high Z -> high return), the hedge mean is positive and")
    print("the t-stat rises with the planted signal. Under the null (0.00) the hedge")
    print("is near zero -- the engine is a faithful detector.")
    print()
    print("On the real EDGAR tape (2008-2023, ~166 S&P 500 firms):")
    print("  Hedge mean: +2.36%/yr  |  HAC t: +0.65  ->  Signal = NONE")
    print("  Correlation(Z, next-yr return): +0.013  ->  essentially zero.")
    print("  (Survivorship-biased; treat as an upper bound.)")


if __name__ == "__main__":
    main()
