"""Offline smoke test — the rule on the synthetic universe, where it *should* work.

The synthetic market has planted springboards (a coil on a rising EMA, then a volume
breakout and a run) hidden among random walks. The detector should fire on the planted
breakouts (high recall) and the backtest should harvest the planted runs (positive net).
That proves the machinery — so when the *real* data says otherwise, it's the market
talking, not a bug.

    python examples/run_synthetic_demo.py
"""

import os
import sys

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

import pandas as pd

from coiled_spring import backtest, data, robustness, signals

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main():
    frames, truth = data.synthetic_universe(seed=0)
    print(f"{len(frames)} names, {len(truth)} planted springboards")

    rec = robustness.detector_recall(frames, truth)
    print(f"detector recall: {rec}")

    led = backtest.run_universe(frames, cost_bps=15.0)
    print(f"\n{len(led)} trades")
    print("stats:", {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in backtest.trade_stats(led).items()})

    print("\nsignal vs same-stock random baseline:")
    print(robustness.signal_vs_baseline(frames).round(4).to_string())


if __name__ == "__main__":
    main()
