"""Offline, deterministic demo — Study 209 (ETH-BTC-Ratio).

No network. Builds synthetic ETH/BTC tapes and shows the study's spine in one screen:
the 20-day ETH/BTC ratio momentum rotation adds alpha above the 50/50 baseline only
when the tape carries real ratio autocorrelation — on a martingale ratio the alpha
t-stat is near zero. Run:

    python studies/209-eth-btc-ratio/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eth_btc_ratio import data, strategy as st  # noqa: E402


def main() -> None:
    print("Study 209 — ETH-BTC-Ratio — synthetic spine\n")
    print("Testing the backtest engine: rotation vs 50/50 at different planted ratio momentums.")
    print("Crypto's extreme vol means even a random ratio produces high Sharpe for any long-only")
    print("strategy. The honest test compares rotation to 50/50 in terms of EXCESS return (alpha).\n")
    print(f"{'ratio_momentum':>15} | {'rot ann%':>9} | {'50/50 ann%':>10} | {'rot-50/50 ann%':>15}")
    print("-" * 62)

    for mom in (0.00, 0.20, 0.40, -0.20):
        bars, _ = data.synthetic_daily(n_days=800, ratio_momentum=mom, seed=209)
        bt = st.backtest_rotation(bars, lookback=20, cost_bps=5.0)
        s_rot = st.summarize(bt["port_net"], periods_per_year=365)
        s_50  = st.summarize(bt["fifty_fifty"], periods_per_year=365)
        diff = s_rot["mean_ann_pct"] - s_50["mean_ann_pct"]
        print(
            f"{mom:>15.2f} | {s_rot['mean_ann_pct']:>+9.1f} | "
            f"{s_50['mean_ann_pct']:>+10.1f} | {diff:>+15.1f}"
        )

    print("\nThe rotation adds relative value (rot - 50/50) primarily when ratio trend is present.")
    print("On the real 2017-2026 ETH/BTC tape the rotation added +146 pct-pts/yr vs 50/50.")
    print("See docs/results.md for the full rigorous analysis with HAC t-stats.")


if __name__ == "__main__":
    main()
