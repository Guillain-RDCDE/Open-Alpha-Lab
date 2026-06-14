"""Offline, deterministic demo — Study 118 (Fed-Model).

No network. Builds a synthetic monthly panel and shows the study's spine in one screen:
the OLS regression finds a real slope and a positive OOS R² *only* when predictability
is planted (predicts > 0), and returns approximately zero when it isn't (the null in a
bottle). Run:

    python studies/118-fed-model/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fed_model import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 118 - Fed-Model - synthetic positive control\n")
    print(
        f"{'predicts':>10} | {'slope':>8} {'R2':>6} {'HAC_t':>7} | "
        f"{'OOS_R2':>8} | finds_signal?"
    )
    print("-" * 70)

    for predicts in (0.0, 0.5, 1.0, 1.5, -0.5):
        panel, truth = data.synthetic_panel(n_months=1_200, predicts=predicts, seed=118)
        sub = panel.dropna(subset=["fed_spread", "fwd10"])
        res = st.ols_hac(sub["fed_spread"].values, sub["fwd10"].values, lags=120)
        oos = st.oos_r2(panel, "fed_spread", "fwd10", train_end_year=1950)

        found = "yes" if res["tstat"] > 2.0 else "no"
        print(
            f"{predicts:>10.2f} | {res['slope']:>+8.3f} {res['r2']:>6.3f} "
            f"{res['tstat']:>+7.2f} | {oos['oos_r2']:>+8.3f} | {found}"
        )

    print()
    print("The regression is a faithful signal detector: it finds the slope when")
    print("predictability is planted (predicts > 0) and returns ~zero on the null.")
    print("On the real Shiller tape the in-sample t is large but OOS R2 < 0 -- see")
    print("docs/results.md for the real-tape verdict.")


if __name__ == "__main__":
    main()
