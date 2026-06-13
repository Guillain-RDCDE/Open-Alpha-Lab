"""Offline two-sided control for Study 100 (Melting-Ice) -- no network.

Shows the harness reports the *sign* of the leverage gap correctly on tapes where we
know the answer:

- ``flat_highvol`` (zero drift, high vol): a 3x sleeve must DECAY -- the realised levered
  NAV ends below the naive "3x the period return" line (drag dominates drift).
- ``uptrend_lowvol`` (steady low-vol uptrend): a 3x sleeve must COMPOUND AHEAD -- the
  realised NAV ends above naive 3x (the trend term beats the small drag).

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from melting_ice import data, strategy  # noqa: E402


def main() -> None:
    for path, label in [("flat_highvol", "FLAT high-vol  (decay must win)"),
                        ("uptrend_lowvol", "UPTREND low-vol (compounding must win)")]:
        frame, truth = data.synthetic_path(path=path)
        c = frame["close"]
        rvn = strategy.realized_vs_naive(c, k=3.0, annual_fee=0.0)
        d = strategy.drag_decomposition(c, k=3.0)
        got = "compounding won" if rvn["compounding_won"] else "decay won"
        expect = "compounding" if truth["expect"] == "compound" else "decay"
        ok = "OK" if got.startswith(expect) else "*** MISMATCH ***"
        print(f"\n=== {label} ===")
        print(f"  realised 3x final x{rvn['realized_final']:.4g}   "
              f"naive 3x final x{rvn['naive_final']:.4g}   gap {rvn['gap']:+.4g}")
        print(f"  drift {d['drift_ann']*100:6.1f}%/yr   drag {d['drag_ann']*100:6.1f}%/yr   "
              f"drag_dominates={d['drag_dominates']}")
        print(f"  -> {got}  (expected: {expect})  {ok}")


if __name__ == "__main__":
    main()
