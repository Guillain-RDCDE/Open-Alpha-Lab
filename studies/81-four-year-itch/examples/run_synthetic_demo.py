"""Offline, deterministic demo — Study 81 (Four-Year-Itch).

No network. Builds a synthetic daily S&P 500 tape and shows the study's spine in one
screen: the Presidential-cycle year-3 premium is detectable *only* when we explicitly plant
it, confirming the machinery works — and that the real-tape verdict is a statement about the
*market*, not the method. Run:

    python studies/81-four-year-itch/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from four_year_itch import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 81 — Four-Year-Itch — synthetic positive control\n")
    print(
        f"{'scenario':>28} | {'y1%':>7} {'y2%':>7} {'y3%':>7} {'y4%':>7} | "
        f"{'y3-rest%':>9} {'Welch-t':>8} | signal?"
    )
    print("-" * 90)

    scenarios = [
        ("null (no cycle)",           (  0.0,   0.0,   0.0,   0.0)),
        ("planted +15 bps/day yr-3",  ( -3.0,  -3.0,  15.0,   3.0)),
        ("planted equal all years",   (  5.0,   5.0,   5.0,   5.0)),
        ("planted Hirsch-like",       ( -6.0,  -3.0,  10.0,   4.0)),
    ]
    for label, premia in scenarios:
        df, _ = data.synthetic_daily(n_cycles=24, year_premia=premia, seed=81)
        s = st.summarize(df)
        print(
            f"{label:>28} | "
            f"{s['y1_mean']:>+7.1f} {s['y2_mean']:>+7.1f} "
            f"{s['y3_mean']:>+7.1f} {s['y4_mean']:>+7.1f} | "
            f"{s['y3_minus_rest_pct']:>+9.1f} {s['y3_vs_rest_t']:>+8.2f} | "
            f"{'YES' if abs(s['y3_vs_rest_t']) >= 2.0 else 'no'}"
        )

    print()
    print("The engine detects a planted cycle when it exists and reads 'no' otherwise.")
    print("On the real S&P 500 tape (~24 cycles back to 1928), see docs/results.md.")


if __name__ == "__main__":
    main()
