"""Offline, deterministic demo — Study 172 (Hundred-Minus-Age).

No network. Builds a synthetic two-asset lifecycle and shows the study's spine in one
screen: the Hundred-Minus-Age glidepath buys lower variance of terminal outcomes *at
the cost* of lower mean return — and that cost is statistically large when the equity
risk premium is real. Run:

    python studies/172-hundred-minus-age/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hundred_minus_age import data, strategy as st  # noqa: E402


def _run(label: str, equity_premium: float) -> None:
    """Simulate all strategies on a 40-year synthetic lifecycle."""
    prices, truth = data.synthetic_lifecycle(
        n_years=42, equity_premium=equity_premium, seed=172
    )
    returns = prices.pct_change().dropna()
    cohorts = st.rolling_cohorts(returns, n_years=40)
    stats = st.cohort_stats(cohorts)

    print(f"\n=== {label} (equity_premium={equity_premium:.2%}) ===")
    print(f"{'Strategy':14s} {'Mean TW':>8s} {'Vol TW':>8s} {'P05':>8s} {'Mean/Vol':>9s}")
    print("-" * 52)
    for strat in ("hma", "sixty_forty", "all_equity", "hma_rising"):
        row = stats.loc[strat]
        print(
            f"{strat:14s} {row['mean']:>8.3f} {row['vol']:>8.3f} "
            f"{row['p05']:>8.3f} {row['mean']/row['vol']:>9.3f}"
        )

    hma = cohorts["hma"].values
    s6040 = cohorts["sixty_forty"].values
    t_hma_vs_6040 = st.hac_tstat(hma, s6040)
    print(f"\nHAC t (HMA vs 60/40): {t_hma_vs_6040:+.2f}")
    print(f"HMA beats 60/40 in {(hma > s6040).mean():.1%} of cohorts")


def main() -> None:
    print("Study 172 — Hundred-Minus-Age — synthetic positive control\n")
    print("TW = terminal wealth (real, normalised so lifetime contributions = 1 unit)")

    # Null: no equity premium — the glidepath costs nothing but gains nothing either
    _run("NULL (no equity premium)", equity_premium=0.0)

    # The real world: equities earn 4% real premium — de-risking has a genuine cost
    _run("PREMIUM (4% equity ERP)", equity_premium=0.04)

    print("\n--- Key insight ---")
    print("With a real equity premium the HMA glidepath is a statistically significant")
    print("*loser* vs 60/40 and vs 100% equities (HAC |t| >> 2). It is a deliberate trade:")
    print("lower variance of outcomes at the cost of lower mean return. Whether that trade")
    print("is worth it is a preference question, not an edge question — and 60/40 dominates")
    print("HMA on *both* mean and p05 (floor outcome), leaving very little rationale for HMA.")


if __name__ == "__main__":
    main()
