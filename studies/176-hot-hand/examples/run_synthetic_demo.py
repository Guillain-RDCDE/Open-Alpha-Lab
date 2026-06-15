"""Offline, deterministic demo — Study 176 (Hot-Hand).

No network. Builds synthetic daily tapes with known autocorrelation and
shows the study's spine in one screen: the hot-hand continuation rate only
exceeds 0.5 when the tape carries real persistence — and on a martingale
(autocorr = 0) it is a fair coin. Run:

    python studies/176-hot-hand/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hot_hand import data, strategy as st  # noqa: E402


def _mean_continuation(bars, max_streak=3):
    """Average continuation rate across all streak lengths 1..max_streak."""
    tbl = st.streak_table(bars["close"], max_streak=max_streak, cost_bps=0.0)
    return float(tbl["continuation"].mean())


def main() -> None:
    print("Study 176 — Hot-Hand — synthetic positive control\n")
    print("Does a streak predict next-day direction?  Planted autocorrelation sweep:\n")
    print(f"{'autocorr':>10} | {'mean cont. rate':>16} | {'hot-hand bps':>12} {'t':>6} | "
          f"{'gambler bps':>11} {'t':>6} | {'interpretation':}")
    print("-" * 90)

    for autocorr in (-0.20, -0.10, 0.00, 0.10, 0.20):
        bars, _ = data.synthetic_daily(n_days=3000, autocorr=autocorr, seed=176)
        cont = _mean_continuation(bars)

        hot = st.summarize(
            st.trade_streak(bars["close"], min_streak=2, direction=0,
                             follow_streak=True, cost_bps=0.0),
            "ret_gross",
        )
        gamb = st.summarize(
            st.trade_streak(bars["close"], min_streak=2, direction=0,
                             follow_streak=False, cost_bps=0.0),
            "ret_gross",
        )

        if autocorr > 0.05:
            interp = "hot-hand is right (momentum)"
        elif autocorr < -0.05:
            interp = "gambler is right (reversion)"
        else:
            interp = "fair coin — neither wins"

        print(
            f"{autocorr:>10.2f} | {cont:>16.3f} | "
            f"{hot['mean_bps']:>+12.2f} {hot['tstat']:>+6.2f} | "
            f"{gamb['mean_bps']:>+11.2f} {gamb['tstat']:>+6.2f} | {interp}"
        )

    print()
    tbl_coin = st.streak_table(
        data.synthetic_daily(n_days=3000, autocorr=0.0, seed=176)[0]["close"],
        max_streak=st.MAX_STREAK,
    )
    bf = st.bonferroni_summary(tbl_coin)
    print(
        f"Bonferroni check on the martingale tape: "
        f"{bf['sig_raw']} cells significant at alpha=0.05 raw, "
        f"{bf['sig_bonf']} after Bonferroni ({bf['n_tests']} tests, "
        f"alpha_adj={bf['alpha_bonf']:.4f})."
    )
    print()
    print("The hot-hand is a faithful momentum meter: it only finds signal when signal exists.")
    print("On a martingale, neither believers win — streaks are just coin-flip memory.")
    print("On the real S&P 500 tape — see docs/results.md — the verdict is NONE.")


if __name__ == "__main__":
    main()
