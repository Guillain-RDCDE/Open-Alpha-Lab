"""Offline positive/negative control for Study 93 (Round-Numbers) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``magnet = 0`` (a constant-drift random walk): price drifts through the round levels with
  no special behaviour, so the fade must NOT bank an edge — its expected return is zero.
- ``magnet > 0`` (a reflective barrier at each round level): price reverses away from the
  round numbers, so the fade should now bank a positive, significant edge — the positive
  control that proves the harness can detect a planted reversal when one exists.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from round_numbers import data, strategy  # noqa: E402

GRID = 100.0
BAND = 0.006
SEED = 93


def _run(magnet: float) -> dict:
    frame, truth = data.synthetic_daily(magnet=magnet, grid=GRID, band=BAND, seed=SEED)
    close = frame["close"]
    tr = strategy.fade_returns(close, GRID, band=BAND, horizon=1)
    return {"truth": truth, "trades": tr, "mean": float(tr.mean()), "t": strategy.hac_tstat(tr)}


def main() -> None:
    for magnet, label in [(0.0, "NULL  (random walk, no magnetism)"),
                          (0.8, "PLANTED magnetism")]:
        r = _run(magnet)
        print(f"\n=== {label} | frac_in_band={r['truth']['frac_in_band']:.2f} ===")
        print(f"  fade : trades {len(r['trades']):4}  mean {r['mean']*1e4:+.2f} bps  "
              f"HAC t {r['t']:+.2f}")
        banks = r["t"] > 2.0
        print(f"  -> fade {'banks' if banks else 'does NOT bank'} the edge (expected: "
              f"{'banks' if magnet > 0 else 'does not bank'})")


if __name__ == "__main__":
    main()
