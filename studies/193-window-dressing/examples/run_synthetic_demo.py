"""Offline, deterministic demo — Study 193 (Window-Dressing).

No network. Builds synthetic daily tapes and shows the study's spine in one screen:
the calendar signal (last N days of each quarter = pump, first N days = reversal) only
beats a random-day placebo when the window-dressing effect is actually planted in the
tape — on a null random-walk it is a fair coin. Run:

    python studies/193-window-dressing/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from window_dressing import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 193 — Window-Dressing — synthetic positive control\n")
    print(
        f"{'planted WD effect':>20} | {'pump bps':>9} {'rev bps':>9} "
        f"{'spread bps':>11} {'t(spread)':>10} | beats random?"
    )
    print("-" * 82)

    for wd in (0.00, 0.50, 1.00, 1.50, -0.50):
        df, truth = data.synthetic_daily(n_years=50, wd_effect=wd, seed=193)
        pmr = st.pump_minus_reversal(df)
        ctrl = st.random_day_control(
            df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=0
        )
        beats = "yes" if pmr["mean_spread_bps"] > ctrl["mean_bps"] + 0.5 else "no"
        print(
            f"{wd:>20.2f} | {pmr['mean_pump_bps']:>+9.2f} {pmr['mean_reversal_bps']:>+9.2f} "
            f"{pmr['mean_spread_bps']:>+11.2f} {pmr['tstat']:>+10.2f} | {beats}"
        )

    print(
        "\nThe calendar is a faithful WD detector: it beats a random-day coin only"
        "\nwhen a pump-and-reversal pattern is planted. On the real SPY tape (wd=0"
        "\neffective), the spread is negative and statistically flat — see docs/results.md."
    )


if __name__ == "__main__":
    main()
