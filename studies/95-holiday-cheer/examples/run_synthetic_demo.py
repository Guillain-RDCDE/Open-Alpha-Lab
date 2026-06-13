"""Offline positive/negative control for Study 95 (Holiday-Cheer) - no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``planted=True``  : an engineered tape with a real return bump on every derived
  pre-holiday day. The harness must DETECT it (large gap, HAC t well above 2).
- ``planted=False`` : a flat i.i.d. tape with engineered holidays but no special
  pre-holiday behaviour. The harness must NOT find a significant effect.

It also demonstrates the holiday-from-gaps derivation recovering the engineered
calendar (e.g. the weekday occurrences of July 4 and Dec 25).

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from holiday_cheer import data, strategy  # noqa: E402


def main() -> None:
    for planted, label in [(True, "PLANTED pre-holiday bump"), (False, "FLAT iid tape")]:
        frame, truth = data.synthetic_daily(planted=planted, seed=95)
        close = frame["close"]
        is_pre = data.pre_holiday_mask(close.index)
        hols = data.derive_market_holidays(close.index)
        e = strategy.effect_stats(close, is_pre)

        jul4 = sum(1 for h in hols if h.month == 7 and h.day == 4)
        dec25 = sum(1 for h in hols if h.month == 12 and h.day == 25)

        print(f"\n=== {label} | n_days={truth['n_days']}  n_pre={truth['n_pre']} ===")
        print(f"  derived {len(hols)} holidays from gaps "
              f"(July-4 weekdays: {jul4}, Dec-25 weekdays: {dec25})")
        print(f"  pre-holiday mean = {e['mu_pre']*1e4:6.2f} bps   "
              f"rest mean = {e['mu_rest']*1e4:5.2f} bps   "
              f"gap = {e['gap']*1e4:+.2f} bps   HAC t(gap) = {e['t_gap']:+.2f}")
        detected = abs(e["t_gap"]) > 2.0
        expected = "detect" if planted else "no detect"
        print(f"  -> {'DETECTED' if detected else 'not detected'} "
              f"(expected: {expected})")


if __name__ == "__main__":
    main()
