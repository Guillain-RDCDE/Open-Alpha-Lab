"""Offline positive/negative control for Study 96 (New-Year-Pop) — no network.

Shows the harness behaves correctly on tapes where we *know* the answer:

- ``jan_bump = 0`` (no seasonal): the small-minus-large spread has no January effect, so
  the contrast must NOT be significant — the harness must not manufacture one.
- ``jan_bump > 0`` (planted January bump): the small leg earns extra return only in
  January, so the harness must detect a significantly higher January spread — the positive
  control that proves the pipeline can bank a real seasonal.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from new_year_pop import data, strategy  # noqa: E402


def _run(jan_bump: float) -> dict:
    frame, truth = data.synthetic_monthly(n_years=36, jan_bump=jan_bump, seed=96)
    c = strategy.contrast(frame)
    return {"truth": truth, "contrast": c}


def main() -> None:
    for bump, label in [(0.0, "NULL  (no seasonal)"), (0.04, "PLANTED Jan bump (+4%/Jan)")]:
        r = _run(bump)
        c = r["contrast"]
        sig = "SIGNIFICANT" if abs(c["t_contrast"]) >= 2.0 else "not significant"
        print(f"\n=== {label} ===")
        print(f"  Jan-minus-rest contrast = {c['diff']*100:+.2f} pts  HAC t = {c['t_contrast']:+.2f}  -> {sig}")
        print(f"  Jan small>large hit-rate = {c['jan_wins']}/{c['n_jan']} "
              f"({c['jan_hit_rate']*100:.0f}%)")
        expect = "detect" if bump > 0 else "find nothing"
        print(f"  -> expected: {expect}")


if __name__ == "__main__":
    main()
