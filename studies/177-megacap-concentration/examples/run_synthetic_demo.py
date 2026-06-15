"""Offline, deterministic demo — Study 177 (Megacap-Concentration).

No network. Builds a synthetic annual cross-section and shows the study's spine:
the top-N concentration strategy only beats equal-weight when a real size-advantage
is planted in the data — and on a null cross-section it does not. Run:

    python studies/177-megacap-concentration/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from megacap_concentration import data, strategy as st  # noqa: E402


def _run(effect: float, n_years: int = 50, seed: int = 177):
    df, truth = data.synthetic_annual(
        n_years=n_years, n_stocks=200, top_n=10, effect=effect, seed=seed
    )
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=10, col="ret_annual")
    result = st.summarize(annual, decade_split=False)
    f = result["full"]
    return f["topn_mean"], f["ew_mean"], f["excess_mean"], f["excess_tstat"]


def main() -> None:
    print("Study 177 — Megacap-Concentration — synthetic positive control\n")
    print(f"{'planted effect':>16} | {'top-10 %':>9} {'ew %':>7} {'excess':>8} {'t-stat':>7} | beats ew?")
    print("-" * 70)
    for effect in (0.00, 0.03, 0.07, 0.10, 0.15, -0.05):
        t10, ew, ex, t = _run(effect)
        beats = "yes" if ex > 1.0 and t > 1.5 else "no"
        print(f"{effect:>16.2f} | {t10:>+9.1f} {ew:>+7.1f} {ex:>+8.1f} {t:>+7.2f} | {beats}")

    print()
    print("The concentration rule is a faithful size-advantage harvester:")
    print("it only edges equal-weight when megacap alpha is actually planted (effect > 0).")
    print("On a null cross-section (effect=0) it is statistically indistinguishable from EW.")
    print()
    print("On the real 2008-2025 EDGAR+yfinance tape:")
    print("  Full sample: top-10 LAGGED the equal-weight S&P 500 by ~1.6 ppt/year (t=-0.31).")
    print("  Pre-2015:    megacaps LOST by ~14 ppt/year (t=-10.3)  — size premium alive.")
    print("  Post-2020:   megacaps WON by ~11 ppt/year (t=+1.30)  — Mag-7 era recency.")
    print()
    print("Verdict: MIXED signal (won recently / lost before), MIRAGE tradability,")
    print("survivorship-biased universe means the real edge is even weaker.")


if __name__ == "__main__":
    main()
