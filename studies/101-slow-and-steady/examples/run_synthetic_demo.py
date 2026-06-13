"""Offline positive/negative control for Study 101 (Slow-and-Steady) — no network.

Shows the harness banks whichever arm is truly ahead, on tapes where we *know* the answer:

- ``drift < 0`` (a down-drifting tape): buying in slowly buys cheaper on average, so
  **DCA should win** terminal wealth on most windows.
- ``drift > 0`` (an up-drifting tape, equities' empirical norm): money put to work sooner
  compounds longer, so **lump-sum should win**.

If the harness banked lump-sum on *both* tapes it would be rigged; banking the correct
arm on each proves it measures the real asymmetry rather than a hard-coded preference.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from slow_and_steady import data, strategy  # noqa: E402


def _run(drift: float) -> dict:
    frame, truth = data.synthetic_daily(n_days=6000, drift=drift, seed=101)
    race = strategy.rolling_race(frame["close"], deploy_months=12, step=5)
    return {"truth": truth, "stats": strategy.win_stats(race)}


def main() -> None:
    for drift, label in [(-0.15, "NEGATIVE drift (DCA should win)"),
                         (+0.15, "POSITIVE drift (lump-sum should win)")]:
        r = _run(drift)
        s = r["stats"]
        winner = "lump_sum" if s["ls_win_rate"] > 0.5 else "dca"
        expected = r["truth"]["expected_winner"]
        print(f"\n=== {label} ===")
        print(f"  lump-sum win-rate : {s['ls_win_rate']*100:5.1f}%  "
              f"mean gap (LS-DCA) {s['mean_gap_pct']:+.2f} cents/$1")
        print(f"  dispersion ratio  : {s['dispersion_ratio']:.3f} (DCA std / LS std)")
        ok = "OK" if winner == expected else "MISMATCH"
        print(f"  -> harness banks '{winner}' (expected '{expected}')  [{ok}]")


if __name__ == "__main__":
    main()
