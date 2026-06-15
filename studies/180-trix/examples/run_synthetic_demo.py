"""Offline, deterministic demo — Study 180 (TRIX).

No network.  Builds synthetic daily tapes and shows the study's spine in one screen:
the TRIX zero-line cross only beats a random-direction coin when the tape actually carries
bar-level momentum — and on a martingale (momentum = 0) it is a fair die.  Run:

    python studies/180-trix/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trix import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 180 — TRIX — synthetic positive control\n")
    print(f"{'planted momentum':>18} | {'n entries':>9} | {'cross bps':>9} {'win':>6} "
          f"{'t':>6} | {'random bps':>10} {'t':>6} | cross beats coin?")
    print("-" * 88)
    for mom in (0.00, 0.08, 0.15, 0.20, -0.10):
        bars, _ = data.synthetic_daily(n_days=1500, momentum=mom, seed=180)
        tx = st.trix(bars["close"], period=15)
        entries = st.zero_line_entries(tx)
        if len(entries) < 5:
            print(f"{mom:>18.2f} | {'<5 entries':>9} | skipping")
            continue
        cross = st.summarize(
            st.run_trades(bars, entries, hold_days=20, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, entries, hold_days=20, cost_bps=0,
                directions=st.random_directions(len(entries), seed=1),
            ),
            "ret_gross",
        )
        beats = "yes" if cross["mean_bps"] > rand["mean_bps"] + 0.0 else "no"
        print(f"{mom:>18.2f} | {len(entries):>9d} | {cross['mean_bps']:>+9.1f} "
              f"{cross['win_rate']:>6.3f} {cross['tstat']:>+6.2f} | "
              f"{rand['mean_bps']:>+10.1f} {rand['tstat']:>+6.2f} | {beats}")

    print()
    print("Signal-line cross (TRIX vs its 9-day EMA) — same sweep:")
    print(f"{'planted momentum':>18} | {'n entries':>9} | {'cross bps':>9} {'win':>6} "
          f"{'t':>6} | {'random bps':>10}")
    print("-" * 88)
    for mom in (0.00, 0.15, -0.10):
        bars, _ = data.synthetic_daily(n_days=1500, momentum=mom, seed=180)
        tx = st.trix(bars["close"], period=15)
        sig = st.trix_signal(tx, signal_period=9)
        entries = st.signal_line_entries(tx, sig)
        if len(entries) < 5:
            print(f"{mom:>18.2f} | {'<5 entries':>9} | skipping")
            continue
        cross = st.summarize(
            st.run_trades(bars, entries, hold_days=10, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(
                bars, entries, hold_days=10, cost_bps=0,
                directions=st.random_directions(len(entries), seed=1),
            ),
            "ret_gross",
        )
        print(f"{mom:>18.2f} | {len(entries):>9d} | {cross['mean_bps']:>+9.1f} "
              f"{cross['win_rate']:>6.3f} {cross['tstat']:>+6.2f} | "
              f"{rand['mean_bps']:>+10.1f}")

    print()
    print("NOTE: With only ~36 zero-line crosses per 1500-day tape, the synthetic control has")
    print("low power.  Both cross and coin are noisy.  The cross beats the coin when momentum")
    print("is planted (cross > rand for positive momentum), confirming the engine's direction.")
    print("On the real daily tape, zero-line cross HAC t=+0.34: no detectable edge — see docs/results.md.")


if __name__ == "__main__":
    main()
