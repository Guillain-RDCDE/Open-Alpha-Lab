"""Offline, deterministic demo — Study 148 (Lunar-Effect).

No network.  Builds a synthetic daily return series with a tunable lunar premium
and shows the study's spine in one screen: the full-vs-new window contrast only
detects the effect when we plant it, and reads near zero on a pure random walk.
Run:

    python studies/148-lunar-effect/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lunar_effect import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 148 — Lunar-Effect — synthetic positive control\n")
    print(f"{'planted discount (full)':>22} | {'new bps':>8} {'full bps':>8} "
          f"{'contrast bps':>13} {'t-contrast':>11} | detects?")
    print("-" * 85)
    for disc in (0.00e-4, 2.0e-4, 5.0e-4, -5.0e-4):
        frame, _ = data.synthetic_lunar(
            n_days=3000, full_discount=disc, new_premium=disc, seed=148
        )
        s = st.summarize(frame)
        detects = "yes" if abs(s["t_contrast"]) > 2.0 else "no"
        print(
            f"{disc * 1e4:>22.1f}bp | "
            f"{s['mean_new_bps']:>8.2f} {s['mean_full_bps']:>8.2f} "
            f"{s['contrast_bps']:>+13.2f} {s['t_contrast']:>+11.2f} | {detects}"
        )

    print()
    print("The contrast is a faithful lunar-premium detector: it fires only when we")
    print("plant a real new-moon premium / full-moon discount.  On the null (disc=0)")
    print("the t-stat is well below ±2.  On the real S&P 500 tape (1928-2026) the")
    print("contrast is +1.1 bps/day at HAC t=0.75 — statistically indistinguishable")
    print("from zero.  See docs/results.md for the full real-tape breakdown.")


if __name__ == "__main__":
    main()
