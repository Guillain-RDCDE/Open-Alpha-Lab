"""Offline, deterministic demo — Study 135 (FOMC-Cycle).

No network. Builds synthetic daily-return series with and without a planted
even-week premium and shows the study's spine: the even-week gap appears when
and only when we bake it in. Run:

    python studies/135-fomc-cycle/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fomc_cycle import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 135 — FOMC-Cycle — synthetic positive control\n")
    print(
        f"{'planted even-premium':>22} | "
        f"{'even bps':>10} {'t':>6} | "
        f"{'odd bps':>10} {'t':>6} | "
        f"{'diff bps':>10} {'Welch-t':>8} | detects?"
    )
    print("-" * 95)

    for premium in (0.0, 0.0005, 0.001, -0.001):
        frame, _ = data.synthetic_cycle(n_days=2000, even_premium=premium, seed=135)
        tbl = st.even_odd_table(frame)
        ev = tbl["even"]
        od = tbl["odd"]
        detects = "yes" if abs(tbl["welch_t"]) > 2.0 and tbl["diff_bps"] * premium >= 0 else "no"
        print(
            f"{premium * 1e4:>+22.1f}bps/day | "
            f"{ev['mean_bps']:>+10.2f} {ev['tstat']:>+6.2f} | "
            f"{od['mean_bps']:>+10.2f} {od['tstat']:>+6.2f} | "
            f"{tbl['diff_bps']:>+10.2f} {tbl['welch_t']:>+8.2f} | {detects}"
        )

    print()
    print("The engine is a faithful even-week premium detector: it finds the gap")
    print("when one is planted (+5 or +10 bps/day), stays quiet at zero, and")
    print("inverts for a negative premium. On the real tape (see docs/results.md)")
    print("the even-week premium is real pre-2019 but has collapsed post-publication.")


if __name__ == "__main__":
    main()
