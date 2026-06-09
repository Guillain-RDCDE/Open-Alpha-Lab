"""Offline demo — prove the machinery on a synthetic mean-reverting panel vs a no-reversion null.

No network, deterministic. On the reversion tape the signal has a real cross-sectional information
coefficient and the book earns gross; on the null, nothing. And on both, inverting the noisy sample
covariance does not beat the naive signal-weighting — the error-maximization the study is about.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from sand_castle import data, statarb, decompose


def run(label, revert):
    panel, market, truth = data.synthetic_panel(n_stocks=50, n_bars=2016, revert=revert, seed=26)
    sq = statarb.signal_quality(panel, market)
    ov = decompose.optimizer_vs_naive(panel, market, cost_bps=5.0)
    print(f"\n=== {label} (revert={revert}, reversion={truth.has_reversion}) ===")
    print(f"  reversion IC {sq['mean_ic']:+.3f} (t {sq['ic_t']:+.1f})")
    print(f"  optimized net Sharpe {ov['optimized_net_sharpe']:+.2f} vs naive net Sharpe "
          f"{ov['naive_net_sharpe']:+.2f} (optimizer minus naive {ov['opt_minus_naive_net']:+.2f})")
    print(f"  turnover optimized {ov['optimized_turnover']:.0f}x/yr")


def main():
    print("Sand-Castle — a synthetic mean-reverting panel vs a no-reversion null.")
    run("REVERSION", revert=0.20)
    run("NULL", revert=0.0)
    print("\nReversion tape: a real IC. Null: nothing. On both, inverting the noisy covariance does not "
          "beat naive signal-weighting. The real S&P 500 verdict is in docs/results.md.")


if __name__ == "__main__":
    main()
