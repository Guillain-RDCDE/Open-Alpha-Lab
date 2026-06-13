"""Offline positive/negative control for Study 102 (Free-Rebalance) — no network.

Shows the harness measures the SIGN of the rebalancing bonus correctly on tapes where
we *know* the answer:

- ``mean_revert``: two equal-drift assets that mean-revert around each other. Rebalancing
  sells the temporary winner and buys the loser — the bonus must be **POSITIVE**
  (Shannon's Demon).
- ``trend``: one asset persistently trends above the other. Rebalancing keeps trimming
  the winner, so the drift (buy-and-hold) portfolio wins — the bonus must be **NEGATIVE**.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from free_rebalance import data, strategy  # noqa: E402


def _run(frame):
    return strategy.rebalancing_bonus(frame, freq="quarterly", cost_bps=10.0)


def main() -> None:
    cases = [
        ("MEAN-REVERT (Shannon's Demon)", data.synthetic_meanrevert(seed=102), +1),
        ("TRENDING asset (bonus is a tax)", data.synthetic_trend(seed=102), -1),
    ]
    for label, (frame, truth), expected in cases:
        r = _run(frame)
        reb, dri = r["rebalanced"], r["drift"]
        sign = "+" if r["bonus_cagr"] >= 0 else "-"
        print(f"\n=== {label} | expected bonus sign {'+' if expected > 0 else '-'} ===")
        print(f"  rebalanced (Q) : CAGR {reb['cagr']*100:6.2f}%  Sharpe {reb['sharpe']:.3f}  maxDD {reb['max_dd']*100:6.1f}%")
        print(f"  drift (B&H)    : CAGR {dri['cagr']*100:6.2f}%  Sharpe {dri['sharpe']:.3f}  maxDD {dri['max_dd']*100:6.1f}%")
        print(f"  -> bonus = {r['bonus_cagr']*100:+.2f} pts/yr CAGR  ({sign}) "
              f"(expected {'+' if expected > 0 else '-'})")
        ok = (r["bonus_cagr"] > 0) == (expected > 0)
        print(f"     sign {'OK' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
