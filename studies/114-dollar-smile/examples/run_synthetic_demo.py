"""Offline, deterministic demo — Study 114 (Dollar-Smile).

No network. Builds a synthetic daily panel and shows the study's spine in one screen:
the dollar-change predictive regression only beats the naive mean when a real link is
planted. On the null tape (pred_r = 0) the regression finds nothing. Run:

    python studies/114-dollar-smile/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dollar_smile import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 114 — Dollar-Smile — synthetic positive control\n")
    print("Testing: does the IS regression find a predictive dollar-equity link?\n")
    print(
        f"{'pred_r':>8} {'contemp_r':>10} | "
        f"{'IS beta SPY':>12} {'IS t':>7} {'OOS R2':>8} | "
        f"{'IS beta EEM':>12} {'IS t':>7} "
    )
    print("-" * 80)

    for pred_r, contemp_r in [
        (0.00, 0.00),
        (-0.30, -0.50),
        (-0.60, -0.80),
        (0.30, 0.50),
    ]:
        daily, _ = data.synthetic_daily(
            n_years=20, pred_r=pred_r, contemp_r=contemp_r, seed=114
        )
        res = st.run_all(daily, horizon="W")
        spy_beta = res["is_spy"]["beta"]
        spy_t = res["is_spy"]["tstat"]
        oos_spy = res["oos_spy"]["r2_oos"] * 100
        eem_beta = res["is_eem"]["beta"]
        eem_t = res["is_eem"]["tstat"]

        print(
            f"{pred_r:>8.2f} {contemp_r:>10.2f} | "
            f"{spy_beta:>+12.3f} {spy_t:>+7.2f} {oos_spy:>+7.2f}% | "
            f"{eem_beta:>+12.3f} {eem_t:>+7.2f}"
        )

    print()
    print("Contemporaneous regression (same-week correlation, not predictive):")
    print(f"{'pred_r':>8} | {'contemp beta SPY':>16} {'t':>7} | {'contemp beta EEM':>16} {'t':>7}")
    print("-" * 60)
    for pred_r, contemp_r in [(0.00, 0.00), (-0.30, -0.50), (-0.60, -0.80)]:
        daily, _ = data.synthetic_daily(
            n_years=20, pred_r=pred_r, contemp_r=contemp_r, seed=114
        )
        res = st.run_all(daily, horizon="W")
        c_spy_t = res["contemp_spy"]["tstat"]
        c_eem_t = res["contemp_eem"]["tstat"]
        c_spy_b = res["contemp_spy"]["beta"]
        c_eem_b = res["contemp_eem"]["beta"]
        print(f"{pred_r:>8.2f} | {c_spy_b:>+16.3f} {c_spy_t:>+7.2f} | {c_eem_b:>+16.3f} {c_eem_t:>+7.2f}")

    print()
    print("The engine is a faithful link detector: IS t grows with planted pred_r,")
    print("and is near-zero on the null tape. The real tape shows: contemporaneous")
    print("correlation is strong (the known coincident link), but IS t ~ 0 and")
    print("OOS R2 < 0 for the predictive regressions -- see docs/results.md.")


if __name__ == "__main__":
    main()
