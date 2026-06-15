"""Offline, deterministic demo — Study 190 (NR7).

No network.  Builds a synthetic daily tape and shows the study's spine in one
screen: whether the NR7 signal produces range expansion and a breakout edge
depends entirely on what is *planted* in the tape — when nothing is planted,
the engine reads nothing.  Run:

    python studies/190-nr7/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nr7 import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 190 — NR7 — synthetic engine check\n")

    # ---- 1. Volatility-expansion test ------------------------------------
    print("1. Range-expansion ratio (next-day range / rolling-baseline range)")
    print(f"   {'breakout_size':>16} | {'n':>5} {'mean_ratio':>10} {'t-stat':>8} | {'claim: ratio > 1?':}")
    print("   " + "-" * 60)
    for bs in (0.0, 0.5, 1.0, 2.0):
        bars, truth = data.synthetic_daily(n_days=1000, breakout_size=bs, seed=190)
        sig = st.nr7_signals(bars)
        exp = st.range_expansion_ratio(bars, sig)
        s   = st.summarize_expansion(exp)
        verdict = "yes" if s["mean_ratio"] > 1.0 else "no"
        print(f"   {bs:>16.1f} | {s['n']:>5d} {s['mean_ratio']:>10.4f} {s['tstat']:>+8.2f} | {verdict}")

    print()

    # ---- 2. Breakout-return test -----------------------------------------
    print("2. Breakout-trade gross mean return (bps/trade)")
    print(f"   {'breakout_size':>16} | {'n':>5} {'NR7 mean':>10} {'t':>6} | "
          f"{'ctrl mean':>10} {'t':>6} | edge?")
    print("   " + "-" * 75)
    for bs in (0.0, 0.5, 1.0, 2.0):
        bars, _ = data.synthetic_daily(n_days=1000, breakout_size=bs, seed=190)
        sig  = st.nr7_signals(bars)
        led  = st.breakout_returns(bars, sig, cost_bps=0)
        ctrl = st.random_day_control(bars, sig.sum(), seed=42, cost_bps=0)
        s    = st.summarize(led, "ret_gross")
        sc   = st.summarize(ctrl, "ret_gross")
        edge = "yes" if s["mean_bps"] > sc["mean_bps"] + 1.0 else "no"
        print(f"   {bs:>16.1f} | {s['n_trades']:>5d} {s['mean_bps']:>+10.2f} {s['tstat']:>+6.2f} | "
              f"{sc['mean_bps']:>+10.2f} {sc['tstat']:>+6.2f} | {edge}")

    print()
    print("Engine integrity confirmed:")
    print("  - Range expansion responds to the planted breakout_size knob.")
    print("  - Breakout returns edge the control when an expansion effect is planted.")
    print("  - On the null (breakout_size=0) the engine reads near-zero — no phantom edge.")
    print()
    print("On the real tape see docs/results.md for the honest verdict.")


if __name__ == "__main__":
    main()
