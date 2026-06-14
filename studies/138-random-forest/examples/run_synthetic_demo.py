"""Offline, deterministic demo — Study 138 (Random-Forest).

No network.  Builds a synthetic daily tape and shows the study's spine in one screen:
the Random Forest walk-forward only beats a shuffled-label control when the tape
actually carries a planted predictable signal — and on a martingale (signal_strength=0)
it matches the shuffle (both hover near 50 %).  Run:

    python studies/138-random-forest/examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from random_forest import data, strategy as st  # noqa: E402


def run_one(signal_strength: float, n_days: int = 1000, seed: int = 138) -> dict:
    bars, _ = data.synthetic_daily(n_days=n_days, signal_strength=signal_strength, seed=seed)
    X, y = st.build_features(bars)
    preds_real = st.walk_forward(
        X, y, train_window=300, step=63, n_estimators=100, max_depth=5, seed=seed
    )
    preds_shuf = st.walk_forward(
        X, y, train_window=300, step=63, n_estimators=100, max_depth=5, seed=seed,
        shuffle_labels=True,
    )
    acc_real = float((preds_real["y_true"] == preds_real["y_pred"]).mean())
    acc_shuf = float((preds_shuf["y_true"] == preds_shuf["y_pred"]).mean())
    ledger = st.trade_predictions(bars, preds_real, cost_bps=5.0)
    stats = st.summarize(preds_real, ledger)
    return {
        "signal_strength": signal_strength,
        "acc_real": acc_real,
        "acc_shuf": acc_shuf,
        "delta_acc": acc_real - acc_shuf,
        "sharpe_net": stats["sharpe_net"],
        "tstat_net": stats["tstat_net"],
    }


def main() -> None:
    print("Study 138 — Random-Forest — synthetic positive control\n")
    print(
        f"{'signal_strength':>16} | {'acc_real':>8} {'acc_shuf':>8} {'delta':>7} | "
        f"{'sharpe_net':>10} {'t_net':>7} | beats shuffle?"
    )
    print("-" * 82)
    for strength in [0.000, 0.005, 0.010, 0.030]:
        r = run_one(strength)
        beats = "yes" if r["delta_acc"] > 0.01 else "no"
        print(
            f"{r['signal_strength']:>16.3f} | {r['acc_real']:>8.3f} {r['acc_shuf']:>8.3f} "
            f"{r['delta_acc']:>+7.3f} | {r['sharpe_net']:>+10.2f} {r['tstat_net']:>+7.2f} "
            f"| {beats}"
        )

    print()
    print("The RF beats the shuffled-label control ONLY when a real signal is planted.")
    print("On a martingale (0.000) it matches the shuffle — both coin-flips at ~50 %.")
    print("On the real SPY tape the result mirrors the null — see docs/results.md.")


if __name__ == "__main__":
    main()
