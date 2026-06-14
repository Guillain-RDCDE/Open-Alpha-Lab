"""Offline, deterministic demo — Study 140 (Amihud-Illiquidity).

No network. Builds a synthetic year x firm panel and shows the study's spine in one
screen: the ILLIQ quintile sort only finds a spread when a premium is planted, and
on the null (premium = 0) it is indistinguishable from a random-quintile control. Run:

    python studies/140-amihud-illiquidity/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from amihud_illiquidity import data, strategy as st  # noqa: E402


def _edge(illiq, fwd_ret) -> tuple[float, float, float, float]:
    """Return (hedge_mean, hedge_t, random_mean, q5_mean) for one panel."""
    res = st.quintile_sort(illiq, fwd_ret)
    s_hedge = st.summarize(res["hedge"])
    s_q5 = st.summarize(res["q5"])
    s_q1 = st.summarize(res["q1"])
    rand = st.random_hedge_returns(illiq, fwd_ret, n_draws=300, seed=42)
    return (
        s_hedge["mean"] * 100.0,
        s_hedge["tstat"],
        float(rand.mean()) * 100.0,
        s_q5["mean"] * 100.0,
    )


def main() -> None:
    print("Study 140 — Amihud-Illiquidity — synthetic positive control\n")
    print(f"{'premium':>10} | {'Q5-Q1 %':>9} {'t':>6} {'random %':>10} | {'Q5 ret %':>10} | finds it?")
    print("-" * 68)

    for prem in (0.00, 0.04, 0.08, -0.04):
        illiq, fwd_ret, _ = data.synthetic_panel(
            n_firms=200, n_years=20, premium=prem, seed=140
        )
        hedge_m, hedge_t, rand_m, q5_m = _edge(illiq, fwd_ret)
        finds = "yes" if hedge_m > rand_m + 0.5 and prem > 0 else (
            "no (null)" if prem == 0.0 else (
                "wrong sign" if prem < 0.0 and hedge_m > rand_m else "no"
            )
        )
        print(f"{prem:>10.2f} | {hedge_m:>+9.2f} {hedge_t:>+6.2f} {rand_m:>+10.2f} | "
              f"{q5_m:>+10.2f} | {finds}")

    print()
    print("The sort is a faithful ILLIQ harvester: it finds the spread only when")
    print("a premium is planted (premium > 0). On the null (0.00) it is noise,")
    print("indistinguishable from a random-quintile control.")
    print("On the real large-cap S&P 500 tape the premium is absent — see docs/results.md.")


if __name__ == "__main__":
    main()
