"""Offline positive/negative control for Study 89 (Turn-of-the-Month) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``bump = 0`` (a flat i.i.d. random walk): there is no turn-of-month cluster, so the
  TOM-minus-rest premium must NOT be significant (|HAC t| should be small).
- ``bump > 0`` (a planted TOM bump): the harness must detect a strongly significant
  TOM-minus-rest premium and recover a TOM-window share of the total return far above
  the ~16% of days that the window occupies.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from turn_of_the_month import data, strategy  # noqa: E402


def _run(bump: float) -> dict:
    frame, truth = data.synthetic_daily(n_days=6000, bump=bump, seed=89)
    close = frame["close"]
    wc = strategy.window_contrast(close, last=1, first=3)
    return {"truth": truth, "wc": wc}


def main() -> None:
    for bump, label in [
        (0.0, "NULL  (flat i.i.d. walk, no TOM cluster)"),
        (0.0010, "PLANTED TOM bump (+10 bps on TOM days)"),
    ]:
        r = _run(bump)
        wc = r["wc"]
        print(f"\n=== {label} | frac_tom_days={r['truth']['frac_tom']:.3f} ===")
        print(f"  TOM mean  {wc['tom_mean_bps']:+6.2f} bps/day   "
              f"rest mean {wc['rest_mean_bps']:+6.2f} bps/day")
        print(f"  diff      {wc['diff_bps']:+6.2f} bps/day   HAC t = {wc['t_diff']:+.2f}")
        print(f"  TOM share of total return = {wc['share_of_total_logret']*100:5.1f}%  "
              f"(window is {wc['frac_tom_days']*100:.1f}% of days)")
        detected = abs(wc["t_diff"]) > 2.0
        expected = "detect" if bump > 0 else "no detection"
        print(f"  -> {'DETECTED' if detected else 'not detected'} "
              f"(expected: {expected})")


if __name__ == "__main__":
    main()
