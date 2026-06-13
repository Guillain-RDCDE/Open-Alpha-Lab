"""Offline, deterministic demo — Study 103 (Turtle-Trader).

No network. Builds synthetic daily tapes and shows the study's spine in one screen:
the Donchian channel breakout only beats a random-entry control when the tape actually
carries day-level trending momentum — and on a martingale (trend = 0) it is no better
than luck. Run:

    python studies/103-turtle-trader/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turtle_trader import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 103 — Turtle-Trader — synthetic positive control (System 1: 20/10)\n")
    print(
        f"{'planted trend':>14} | {'n_trades':>8} {'breakout bps':>13} {'t':>6} "
        f"| {'random bps':>10} {'t':>6} | beats random?"
    )
    print("-" * 80)

    for trend in (0.00, 0.10, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=600, trend=trend, seed=103)
        sig     = st.donchian_signals(bars, entry_n=20, exit_n=10)
        led     = st.run_trades(bars, sig, cost_bps=0)

        if len(led) == 0:
            print(f"{trend:>14.2f} | {'(no trades)':>8}")
            continue

        pairs = st.random_entry_pairs(len(led), len(bars), entry_n=20, seed=1)
        rled  = st.run_trades(bars, sig, cost_bps=0, random_entry_dates=pairs)

        bk = st.summarize(led,  "ret_gross")
        rk = st.summarize(rled, "ret_gross")

        beats = "yes" if bk["mean_bps"] > rk["mean_bps"] + 0.5 else "no"
        print(
            f"{trend:>14.2f} | {bk['n_trades']:>8d} {bk['mean_bps']:>+13.2f} "
            f"{bk['tstat']:>+6.2f} | {rk['mean_bps']:>+10.2f} {rk['tstat']:>+6.2f} | {beats}"
        )

    print()
    print("System 2 (55/20) sweep:")
    print(
        f"{'planted trend':>14} | {'n_trades':>8} {'breakout bps':>13} {'t':>6} "
        f"| {'random bps':>10} | beats random?"
    )
    print("-" * 80)

    for trend in (0.00, 0.10, 0.20):
        bars, _ = data.synthetic_daily(n_days=800, trend=trend, seed=103)
        sig     = st.donchian_signals(bars, entry_n=55, exit_n=20)
        led     = st.run_trades(bars, sig, cost_bps=0, max_hold=365)

        if len(led) == 0:
            print(f"{trend:>14.2f} | {'(no trades)':>8}")
            continue

        pairs = st.random_entry_pairs(len(led), len(bars), entry_n=55, seed=1)
        rled  = st.run_trades(bars, sig, cost_bps=0, random_entry_dates=pairs, max_hold=365)

        bk = st.summarize(led,  "ret_gross")
        rk = st.summarize(rled, "ret_gross")

        beats = "yes" if bk["mean_bps"] > rk["mean_bps"] + 0.5 else "no"
        print(
            f"{trend:>14.2f} | {bk['n_trades']:>8d} {bk['mean_bps']:>+13.2f} "
            f"{bk['tstat']:>+6.2f} | {rk['mean_bps']:>+10.2f} | {beats}"
        )

    print()
    print("The Donchian breakout is a trend harvester: it edges random entries only")
    print("when momentum is planted. On a martingale it is no better than luck.")
    print("On the real tape (long history, diverse basket) — see docs/results.md.")


if __name__ == "__main__":
    main()
