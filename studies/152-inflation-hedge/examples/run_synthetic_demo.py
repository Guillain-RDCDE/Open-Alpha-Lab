"""Offline, deterministic demo — Study 152 (Inflation-Hedge).

No network.  Builds a synthetic monthly inflation/equity panel and shows the
study's spine in one screen:

- With ``fisher_beta = 0.0`` (the Fama-Schwert null): nominal returns have NO
  relationship with inflation, real returns fall when inflation rises, and the
  inflation regime predicts lower forward real returns in the high regime.
- With ``fisher_beta = 1.0`` (the full-hedge world): nominal returns rise one-
  for-one with inflation, real returns are approximately invariant, and the
  regime difference disappears.

This confirms the machinery recovers the planted parameter correctly.  On the
*real* Shiller tape (see examples/verify.py), the data look far closer to the
Fama-Schwert null than to the full-hedge world.

Run:
    python studies/152-inflation-hedge/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inflation_hedge import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 152 - Inflation-Hedge - synthetic Fisher beta sweep\n")
    print(
        f"{'planted beta':>14} | {'nom OLS beta':>12} {'nom t':>7} | "
        f"{'real OLS beta':>13} {'real t':>7} | "
        f"{'high vs low 1y (pp)':>20} {'regime t':>9}"
    )
    print("-" * 95)

    for beta in (0.00, 0.25, 0.50, 0.75, 1.00):
        df, _ = data.synthetic_world(n_years=130, fisher_beta=beta, seed=152)
        res = st.summarize(df)
        diff_pct = res["diff_mean_1y"] * 100.0
        print(
            f"{beta:>14.2f} | "
            f"{res['fisher_beta_nom']:>+12.3f} {res['fisher_t_nom']:>+7.2f} | "
            f"{res['fisher_beta_real']:>+13.3f} {res['fisher_t_real']:>+7.2f} | "
            f"{diff_pct:>+20.2f}pp {res['t_diff_1y']:>+9.2f}"
        )

    print()
    print("Fisher beta = 0.0 -> nominal beta ~0, real beta strongly negative,")
    print("  high-inflation regime has lower forward real returns.")
    print("Fisher beta = 1.0 -> nominal beta ~1, real beta ~0, regime difference ~0.")
    print()
    print("On the REAL Shiller tape (1872-2023):")
    print("  nominal Fisher beta ~0.31  (t ~1.30, NOT 1.0)  -> hedge fails")
    print("  real Fisher beta ~-0.72    (t ~-3.02)           -> real returns fall")
    print("  high vs low regime 1y diff ~-7.1pp (t ~-4.08)  -> regime matters")
    print()
    print("VERDICT: Signal=MIXED (real effect real & significant; long-run better)")
    print("         Tradability=MIRAGE (regime not sharp enough to time the market)")


if __name__ == "__main__":
    main()
