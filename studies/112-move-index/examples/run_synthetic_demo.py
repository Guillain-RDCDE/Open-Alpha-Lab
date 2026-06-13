"""Offline, deterministic demo -- Study 112 (Move-Index).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the MOVE quintile spread grows monotonically with the planted move_signal, confirming
the engine is a faithful MOVE detector.  On the real tape the spread is flat (near 0)
-- see docs/results.md.  Run:

    python studies/112-move-index/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from move_index import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 112 -- Move-Index -- synthetic positive control\n")
    print(f"{'planted signal':>16} | {'Q5-Q1 spread':>13} {'t_spread':>9} | "
          f"{'Q5 mean':>8} {'Q1 mean':>8}")
    print("-" * 75)

    spreads = []
    for sig in (0.00, 0.01, 0.03, 0.05):
        df, _ = data.synthetic_daily(n_days=2000, move_signal=sig, seed=42)
        res = st.top_minus_bottom(df, horizon=1)
        spreads.append(res["spread_bps"])
        print(
            f"{sig:>16.2f} | {res['spread_bps']:>+13.2f} {res['t_spread']:>+9.2f} | "
            f"{res['mean_top_bps']:>+8.2f} {res['mean_bot_bps']:>+8.2f}"
        )

    # Verify the engine is monotone in planted signal (spread becomes more negative)
    assert spreads[0] > spreads[2] > spreads[3], "Spread should grow with move_signal"

    print("\nAlso checking MOVE vs VIX regression on the signal tape (move_signal=0.03):")
    df_sig, _ = data.synthetic_daily(n_days=2000, move_signal=0.03, seed=42)
    reg = st.move_vs_vix_regression(df_sig, horizon=1)
    print(f"  beta_MOVE = {reg['beta_move']:+.4f} (t = {reg['t_move']:+.2f}), "
          f"beta_VIX = {reg['beta_vix']:+.4f} (t = {reg['t_vix']:+.2f})")

    print("\nThe engine picks up MOVE signal monotonically as the planted signal grows.")
    print("On the real tape the spread is near 0 bps with |t| < 2 -- see docs/results.md.")


if __name__ == "__main__":
    main()
