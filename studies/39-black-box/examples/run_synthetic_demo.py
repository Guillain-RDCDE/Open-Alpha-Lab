"""Offline demo — the black-box machinery on the synthetic control, no network.

    python examples/run_synthetic_demo.py

Builds the seeded synthetic predictable series (a weak nonlinear lagged signal) and the random-walk null,
fits the MLP, and shows the headline trap: a gorgeous IN-SAMPLE Sharpe that collapses out-of-sample —
even on the null, where there is nothing to learn — plus the shuffled-label proof that the in-sample
number is the net fitting noise. The real-crypto verdict is in ../docs/results.md.
"""

from __future__ import annotations

import os
import sys
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
warnings.filterwarnings("ignore")

from black_box import data, extension, strategy


NET = dict(hidden=(16, 8), max_iter=250)
WF = dict(min_train=400, step=200, **NET)


def main() -> None:
    # Strong-enough control signal so the net visibly RECOVERS it out-of-sample; a random-walk null.
    df, truth = data.synthetic_predictable(predictable_strength=0.015, n_bars=1200, seed=39)
    df0, _ = data.synthetic_predictable(predictable_strength=0.0, n_bars=1200, seed=39)
    close, close0 = df["close"], df0["close"]
    print(f"Synthetic control: {truth.n_bars} days, predictable_strength {truth.predictable_strength} "
          f"(null = 0)\n")

    for nm, c in [("PREDICTABLE control", close), ("NULL (random walk)", close0)]:
        gap = extension.insample_vs_oos(c, cost_bps=10.0, **WF)
        is_sh = gap.loc["in_sample", "sharpe_gross"]
        oos_sh = gap.loc["walk_forward_oos", "sharpe_gross"]
        oos_net = gap.loc["walk_forward_oos", "sharpe_net"]
        print(f"  {nm:22} in-sample Sharpe {is_sh:6.2f}  |  walk-forward OOS {oos_sh:6.2f}  "
              f"(net@10bp {oos_net:6.2f})")

    print("\nThe trap, on the control (in-sample vs walk-forward):")
    print(extension.insample_vs_oos(close, cost_bps=10.0, **WF).round(3).to_string())

    print("\nShuffled-label control on the NULL (train accuracy survives meaningless targets):")
    print(extension.shuffled_label_control(close0, n_shuffles=4, **NET).round(3).to_string())

    print("\nReal-crypto verdict (BTC/ETH/LTC/XRP) is in ../docs/results.md.")


if __name__ == "__main__":
    main()
