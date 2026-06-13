"""Offline positive/negative control for Study 88 (Dogs-of-the-Dow) -- no network.

Shows the harness behaves correctly on panels where we *know* the answer:

- ``planted=True``  -- high trailing yield genuinely forecasts a higher forward total
  return, so the top-ten-yield basket MUST beat the equal-weight index (a positive HAC t).
- ``planted=False`` -- yield is drawn independently of forward return, so the basket must
  NOT beat the index (a t indistinguishable from zero).

A pipeline that cannot bank the planted edge proves nothing by finding nothing on the real
tape; a pipeline that 'finds' an edge in the null is broken. This is that machinery proof.

    python examples/run_synthetic_demo.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from dogs_of_the_dow import data, strategy  # noqa: E402


def _run(planted: bool) -> dict:
    panel = data.synthetic_panel(planted=planted, seed=88)
    ann = strategy.annual_returns_from_synthetic(panel)
    s = strategy.summarize(ann)
    t = strategy.hac_tstat(ann["excess"].to_numpy())
    return {"truth": panel["truth"], "summary": s, "t": t}


def main() -> None:
    for planted, label in [(True, "PLANTED (yield forecasts return)"),
                           (False, "NULL    (yield is noise)")]:
        r = _run(planted)
        s, t = r["summary"], r["t"]
        print(f"\n=== {label} | planted excess = {r['truth']['excess']*100:.1f}% ===")
        print(f"  mean annual excess = {s['mean_excess']*100:+.2f}%/yr   "
              f"win rate {s['win_rate']*100:.0f}%   HAC t = {t:+.2f}")
        detected = abs(t) > 2
        expected = planted
        ok = detected == expected
        print(f"  -> edge detected: {detected} (expected: {expected})  "
              f"[{'OK' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
