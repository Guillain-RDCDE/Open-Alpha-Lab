"""Offline positive/negative control for Study 90 (Weekend / day-of-week effect) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``dow_effect = 0`` (an i.i.d. constant-drift tape): there is no weekday structure, so the
  per-weekday contrasts must NOT clear the bar and a "buy Tuesday" timer must not out-earn
  buy-and-hold on a risk-adjusted basis.
- ``dow_effect > 0`` (Monday down, Tuesday up, by a known bump): the contrasts must now light
  up — Monday-vs-rest significantly negative, Tuesday-vs-rest significantly positive — the
  positive control that proves the harness can detect a planted seasonality when one exists.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from weekend import data, strategy  # noqa: E402


def _run(dow_effect: float) -> dict:
    frame, truth = data.synthetic_daily(n_days=6000, dow_effect=dow_effect, seed=90)
    close = frame["close"]
    return {
        "truth": truth,
        "mon": strategy.contrast(close, 0),
        "tue": strategy.contrast(close, 1),
    }


def main() -> None:
    for eff, label in [(0.0, "NULL  (i.i.d., no weekday structure)"),
                       (0.0010, "PLANTED weekend effect (Mon -10bps / Tue +10bps)")]:
        r = _run(eff)
        mon, tue = r["mon"], r["tue"]
        print(f"\n=== {label} ===")
        print(f"  Monday  - rest: {mon['diff_bps']:+7.2f} bps   HAC t {mon['hac_t']:+5.2f}")
        print(f"  Tuesday - rest: {tue['diff_bps']:+7.2f} bps   HAC t {tue['hac_t']:+5.2f}")
        detected = (mon["hac_t"] < -2) and (tue["hac_t"] > 2)
        expected = eff > 0
        verdict = "DETECTED" if detected else "not detected"
        print(f"  -> seasonality {verdict} (expected: {'yes' if expected else 'no'})")


if __name__ == "__main__":
    main()
