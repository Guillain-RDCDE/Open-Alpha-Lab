"""Offline, deterministic demo — Study 201 (Dividend-Growth).

No network.  Builds a synthetic annual panel and shows the study's spine in one screen:
the dividend-growth screen outperforms the equal-weight basket only when a return premium is
planted in the data — on a null tape (no premium) it is indistinguishable from the basket.
Run:

    python studies/201-dividend-growth/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_growth import data, strategy as st  # noqa: E402


def one_run(growth_premium: float, seed: int = 201) -> dict:
    p = data.synthetic_panel(growth_premium=growth_premium, seed=seed)
    mask = st.classify_growers(p["streaks"], streak_years=3)
    grower_ret = st.arm_returns(p["fwd_ret"], mask)
    ew_ret = p["ew_ret"].reindex(grower_ret.index)
    spread = st.hedge_returns(grower_ret, ew_ret)
    s_g = st.summary(grower_ret)
    s_ew = st.summary(ew_ret)
    s_sp = st.summary(spread)
    return {
        "premium": growth_premium,
        "grower_mean": s_g["mean"],
        "ew_mean": s_ew["mean"],
        "spread_mean": s_sp["mean"],
        "spread_t": s_sp["tstat"],
        "n": s_g["n"],
    }


def main() -> None:
    print("Study 201 — Dividend-Growth — synthetic positive control\n")
    print(f"{'planted prem':>14} | {'grower':>8} {'EW':>8} {'spread':>8} {'t':>6} | signal detectable?")
    print("-" * 72)
    for prem in (0.00, 0.02, 0.05, 0.08):
        r = one_run(prem)
        detectable = "yes" if r["spread_t"] > 2.0 else "no"
        print(
            f"{prem:>14.2f} | {r['grower_mean']:>+8.1%} {r['ew_mean']:>+8.1%} "
            f"{r['spread_mean']:>+8.1%} {r['spread_t']:>+6.2f} | {detectable}"
        )

    print()
    print("The grower screen is a faithful premium detector: it finds the edge when a")
    print("premium is planted (>=5%) and shows near-zero spread on the null (0%).")
    print("Whether the real tape has any premium to harvest — see docs/results.md.")


if __name__ == "__main__":
    main()
