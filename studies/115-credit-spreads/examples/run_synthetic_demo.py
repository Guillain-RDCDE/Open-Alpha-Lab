"""Offline, deterministic demo — Study 115 (Credit-Spreads).

No network.  Builds a synthetic daily price frame and shows the study's spine in one
screen: the credit-spread signal only predicts negative equity returns when the effect
is actually planted.  On a null tape (spread_signal=0) the strategy is a fair coin.

    python studies/115-credit-spreads/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from credit_spreads import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 115 -- Credit-Spreads -- synthetic positive control\n")
    print(
        f"{'spread_signal':>14} | "
        f"{'corr(spread->SPY)':>18} {'diff bps':>9} {'diff t':>7} | "
        f"signal found?"
    )
    print("-" * 72)
    for sig in (0.0, 0.5, 1.0, 2.0, -1.0):
        prices, truth = data.synthetic_daily(n_days=2000, spread_signal=sig, seed=115)
        result = st.summarize(prices, window=20, forward_days=5, rolling_median_window=60)
        corr = result["spread_corr"]
        diff_bps = result["spread_diff_mean_bps"]
        diff_t = result["spread_diff_tstat"]
        # With enough data and strong signal, the correlation should be negative
        found = "YES" if abs(corr) > 0.05 else "no"
        print(
            f"{sig:>14.1f} | "
            f"{corr:>18.4f} {diff_bps:>9.1f} {diff_t:>7.2f} | "
            f"{found}"
        )

    print()
    print("The negative correlation (spread widening -> lower forward SPY)")
    print("grows with the planted signal strength.")
    print("On the real tape (n~4000) the effect is present but weak -- see docs/results.md.")


if __name__ == "__main__":
    main()
